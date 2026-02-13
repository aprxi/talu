//! BPE (Byte-Pair Encoding) Model
//!
//! Implements BPE tokenization with lazy vocab/merges parsing.
//! Supports both GPT-2 style (byte-level) and SentencePiece BPE.

const std = @import("std");
const ct = @import("c_types.zig");
const json_utils = @import("json_utils.zig");
const utils = @import("utils.zig");
const log = @import("../log.zig");

const tok_fns = @import("pipeline.zig");

// Internal helpers from utils
const utf8Encode = utils.utf8Encode;
const utf8Decode = utils.utf8Decode;
const utf8CharLen = utils.utf8CharLen;

const DEFAULT_PATTERN: [:0]const u8 = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
const DEFAULT_UNK: [:0]const u8 = "<unk>";

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

    // Merges: hash map for O(1) lookup
    merges: std.StringHashMapUnmanaged(i32),
    merge_strings: std.ArrayListUnmanaged([]const u8),

    // Byte mapping (small, always built)
    byte_to_unicode: [256][]const u8,
    unicode_to_byte: [65536]i32,

    // SentencePiece byte fallback: maps byte value -> token ID for <0xNN> tokens
    // Used when a character can't be found in vocab (e.g., control chars like \n)
    byte_fallback_ids: [256]i32,

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
    const vocab_size = findVocabSize(json_buffer);

    model.* = .{
        .allocator = allocator,
        .json_buffer = json_buffer,
        .json_owned = json_owned,
        .id_to_token = try allocator.alloc(?[]const u8, vocab_size),
        .vocab_hash = .{},
        .merges = .{},
        .merge_strings = .{},
        .byte_to_unicode = undefined,
        .unicode_to_byte = [_]i32{-1} ** 65536,
        .byte_fallback_ids = [_]i32{-1} ** 256,
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
fn findVocabSize(json: []const u8) usize {
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
                    num = num * 10 + (json[scan_idx] - '0');
                    scan_idx += 1;
                }
                if (num > max_id) max_id = num;
            }
        } else {
            scan_idx += 1;
        }
    }
    return max_id + 1;
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


const EncodedWord = struct {
    ids: []i32,
    tokens: [][*:0]u8,
};

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

fn encodeWord(model: *BpeModel, tok: *ct.Tokenizer, word: []const u8) !EncodedWord {
    const allocator = model.allocator;
    log.trace("tokenizer", "encodeWord", .{ .word_len = word.len, .vocab_size = model.vocab_hash.count() }, @src());
    const word_z = try allocator.dupeZ(u8, word);
    defer allocator.free(word_z);

    if (tok_fns.tokenizer_added_token_find(tok, word_z.ptr)) |added| {
        const ids = try allocator.alloc(i32, 1);
        errdefer allocator.free(ids);
        const toks = try allocator.alloc([*:0]u8, 1);
        errdefer allocator.free(toks);
        ids[0] = added.*.id;
        const dup_tok = try allocator.dupeZ(u8, word);
        toks[0] = dup_tok.ptr;
        return EncodedWord{ .ids = ids, .tokens = toks };
    }

    // Detect if this is byte-level BPE (GPT-2 style) vs SentencePiece style
    // Check the tokenizer's pretokenizer settings
    const is_byte_level = tok.pretokenizer.byte_level != 0;

    // Direct vocab lookup first - works for both SentencePiece and byte-level BPE
    // For byte-level BPE: the pretokenizer already converted bytes to GPT-2 unicode,
    // so the input is already in the correct format for vocab lookup
    // For SentencePiece: tokens like "▁is" are stored as UTF-8 strings directly
    if (model.vocab_hash.get(word)) |id| {
        log.trace("tokenizer", "Direct vocab hit", .{ .id = id }, @src());
        const ids = try allocator.alloc(i32, 1);
        errdefer allocator.free(ids);
        const toks = try allocator.alloc([*:0]u8, 1);
        errdefer allocator.free(toks);
        ids[0] = id;
        const dup_tok = try allocator.dupeZ(u8, word);
        toks[0] = dup_tok.ptr;
        return EncodedWord{ .ids = ids, .tokens = toks };
    }
    log.trace("tokenizer", "Vocab miss, using BPE", .{}, @src());

    var tokens = std.ArrayListUnmanaged([]const u8){};
    defer tokens.deinit(allocator);

    // Split input into initial tokens for BPE merging
    if (is_byte_level) {
        // Byte-level BPE (GPT-2): input is already GPT-2 unicode encoded by pretokenizer
        // Split by UTF-8 characters
        var byte_idx: usize = 0;
        while (byte_idx < word.len) {
            const char_len = utf8CharLen(word[byte_idx]);
            if (byte_idx + char_len > word.len) return error.InvalidUtf8;
            try tokens.append(allocator, word[byte_idx .. byte_idx + char_len]);
            byte_idx += char_len;
        }
    } else {
        // SentencePiece BPE: split by UTF-8 characters directly
        // Then apply BPE merges to respect merge priority order
        var byte_idx: usize = 0;
        while (byte_idx < word.len) {
            const char_len = utf8CharLen(word[byte_idx]);
            if (byte_idx + char_len > word.len) return error.InvalidUtf8;
            try tokens.append(allocator, word[byte_idx .. byte_idx + char_len]);
            byte_idx += char_len;
        }
    }

    var scratch = std.ArrayListUnmanaged(u8){};
    defer scratch.deinit(allocator);

    // Merge loop
    var owned_tokens = std.ArrayListUnmanaged([]u8){}; // track allocations
    defer {
        for (owned_tokens.items) |t| allocator.free(t);
        owned_tokens.deinit(allocator);
    }

    while (true) {
        const best = try findBestPair(tokens.items, &model.merges, &scratch, allocator) orelse break;
        const pos = best.pos;
        const left = tokens.items[pos];
        const right = tokens.items[pos + 1];

        // Create merged token
        const merged = try allocator.alloc(u8, left.len + right.len);
        @memcpy(merged[0..left.len], left);
        @memcpy(merged[left.len..], right);
        try owned_tokens.append(allocator, merged);

        // Update token list
        tokens.items[pos] = merged;
        _ = tokens.orderedRemove(pos + 1);
    }

    // Convert to IDs
    // For SentencePiece, tokens not in vocab need byte fallback (<0xNN>)
    var result_ids = std.ArrayListUnmanaged(i32){};
    errdefer result_ids.deinit(allocator);
    var result_tokens_list = std.ArrayListUnmanaged([*:0]u8){};
    errdefer {
        for (result_tokens_list.items) |t| allocator.free(std.mem.sliceTo(t, 0));
        result_tokens_list.deinit(allocator);
    }

    for (tokens.items) |token| {
        if (model.vocab_hash.get(token)) |id| {
            // Token found in vocab
            try result_ids.append(allocator, id);
            const dup = try allocator.dupeZ(u8, token);
            try result_tokens_list.append(allocator, dup.ptr);
        } else if (!is_byte_level) {
            // SentencePiece: use byte fallback for unknown tokens
            for (token) |byte_val| {
                const fallback_id = model.byte_fallback_ids[byte_val];
                if (fallback_id >= 0) {
                    try result_ids.append(allocator, fallback_id);
                    // Build token string "<0xNN>"
                    var tok_str: [7]u8 = undefined;
                    tok_str[0] = '<';
                    tok_str[1] = '0';
                    tok_str[2] = 'x';
                    const hex_chars = "0123456789ABCDEF";
                    tok_str[3] = hex_chars[byte_val >> 4];
                    tok_str[4] = hex_chars[byte_val & 0x0F];
                    tok_str[5] = '>';
                    tok_str[6] = 0;
                    const dup = try allocator.dupeZ(u8, tok_str[0..6]);
                    try result_tokens_list.append(allocator, dup.ptr);
                } else {
                    // No byte fallback, use UNK
                    try result_ids.append(allocator, model.unk_id);
                    const dup = try allocator.dupeZ(u8, token);
                    try result_tokens_list.append(allocator, dup.ptr);
                }
            }
        } else {
            // Byte-level BPE: use UNK for unknown tokens
            try result_ids.append(allocator, model.unk_id);
            const dup = try allocator.dupeZ(u8, token);
            try result_tokens_list.append(allocator, dup.ptr);
        }
    }

    return EncodedWord{
        .ids = try result_ids.toOwnedSlice(allocator),
        .tokens = try result_tokens_list.toOwnedSlice(allocator),
    };
}

fn freeEncodedWord(allocator: std.mem.Allocator, encoded: EncodedWord) void {
    for (encoded.tokens) |tok_ptr| allocator.free(std.mem.sliceTo(tok_ptr, 0));
    allocator.free(encoded.tokens);
    allocator.free(encoded.ids);
}

// ============= C API callbacks =============

fn bpe_encode(tok: *ct.Tokenizer, input: [*c]const u8, enc: *ct.TokenizerEncoding) c_int {
    if (tok.model == null) return -1;
    const model = @as(*BpeModel, @ptrCast(@alignCast(tok.model.?)));
    const text = std.mem.sliceTo(input, 0);

    log.trace("tokenizer", "bpe_encode", .{ .text_len = text.len, .vocab_size = model.vocab_hash.count() }, @src());

    const encoded = encodeWord(model, tok, text) catch |err| {
        log.trace("tokenizer", "encodeWord failed", .{ .err = @errorName(err) }, @src());
        tok_fns.tokenizer_set_error(tok, "BPE encode failed");
        return -1;
    };

    enc.ids_len = encoded.ids.len;
    enc.tokens_len = encoded.tokens.len;
    enc.ids = @ptrCast(encoded.ids);
    enc.tokens = @ptrCast(encoded.tokens);
    return 0;
}

/// Encode with explicit length (supports embedded null bytes) - implementation
fn bpe_encode_slice_impl(model: *BpeModel, tok: *ct.Tokenizer, text: []const u8, enc: *ct.TokenizerEncoding) c_int {
    log.trace("tokenizer", "bpe_encode_slice", .{ .text_len = text.len, .vocab_size = model.vocab_hash.count() }, @src());

    const encoded = encodeWord(model, tok, text) catch |err| {
        log.trace("tokenizer", "encodeWord failed", .{ .err = @errorName(err) }, @src());
        tok_fns.tokenizer_set_error(tok, "BPE encode failed");
        return -1;
    };

    enc.ids_len = encoded.ids.len;
    enc.tokens_len = encoded.tokens.len;
    enc.ids = @ptrCast(encoded.ids);
    enc.tokens = @ptrCast(encoded.tokens);
    return 0;
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
        // Check id_to_token array (regular vocab tokens are not special)
        if (id >= 0 and @as(usize, @intCast(id)) < model.id_to_token.len) {
            if (model.id_to_token[@as(usize, @intCast(id))]) |token| {
                slot.* = .{ .slice = token, .is_special = false };
                continue;
            }
        }
        // Check added tokens - use the actual special flag from tokenizer config
        if (findAddedTokenInfoById(tok, id)) |added_info| {
            slot.* = .{ .slice = std.mem.sliceTo(added_info.content, 0), .is_special = added_info.is_special };
            continue;
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
                // SentencePiece: U+2581 (▁) represents word boundary, convert to space
                if (codepoint == 0x2581) {
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

    // Strip leading space added by add_prefix_space (pretokenizer or Metaspace decoder)
    if ((tok.pretokenizer.add_prefix_space != 0 or tok.decoder.add_prefix_space != 0) and result.items.len > 0 and result.items[0] == ' ') {
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

    // Free merge strings (we allocated these)
    for (model.merge_strings.items) |s| model.allocator.free(s);
    model.merge_strings.deinit(model.allocator);
    model.merges.deinit(model.allocator);
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

    var model = try create(allocator, json_buffer, json_owned);
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
        return error.PretokenizerInitFailed;
    }

    return tok;
}

// =============================================================================

/// Destroy implementation for file-loaded models (same as lazy but also frees vocab strings)
fn bpe_destroy_files_impl(model: *BpeModel, tok: *ct.Tokenizer) void {
    tok.model = null;

    // Free vocab strings (we allocated these when parsing vocab.json)
    for (model.id_to_token) |maybe_token| {
        if (maybe_token) |token| {
            model.allocator.free(token);
        }
    }

    // Rest is same as lazy destroy
    for (model.merge_strings.items) |s| model.allocator.free(s);
    model.merge_strings.deinit(model.allocator);
    model.merges.deinit(model.allocator);
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
        .merge_strings = .{},
        .byte_to_unicode = undefined,
        .unicode_to_byte = [_]i32{-1} ** 65536,
        .byte_fallback_ids = [_]i32{-1} ** 256,
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
    const size = findVocabSize(json);
    try std.testing.expectEqual(@as(usize, 3), size);
}

test "tokenizer_bpe_create_from_spec: findVocabSize with gaps in IDs" {
    const json =
        \\{"vocab": {"a": 0, "b": 5, "c": 10}}
    ;
    const size = findVocabSize(json);
    try std.testing.expectEqual(@as(usize, 11), size);
}

test "tokenizer_bpe_create_from_spec: findVocabSize with large IDs" {
    const json =
        \\{"vocab": {"token1": 100, "token2": 50000, "token3": 25}}
    ;
    const size = findVocabSize(json);
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
    const size = findVocabSize(json);
    try std.testing.expectEqual(@as(usize, 3), size);
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
        model.vocab_hash.deinit(allocator);
        model.merges.deinit(allocator);
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
    try std.testing.expect(model.vocab_hash.get("a").? == 0);
    try std.testing.expect(model.vocab_hash.get("b").? == 1);
    try std.testing.expect(model.vocab_hash.get("c").? == 2);

    // Verify id_to_token was populated
    try std.testing.expect(model.id_to_token[0] != null);
    try std.testing.expect(model.id_to_token[1] != null);
    try std.testing.expect(model.id_to_token[2] != null);

    // Verify merges were parsed (array format)
    try std.testing.expect(model.merges.count() > 0);
}

test "tokenizer_bpe_create_from_spec: parseVocabAndMerges with string format merges" {
    const json =
        \\{"vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"]}
    ;
    const allocator = std.testing.allocator;

    var model = try create(allocator, json, false);
    defer {
        model.vocab_hash.deinit(allocator);
        model.merges.deinit(allocator);
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
        model.vocab_hash.deinit(allocator);
        model.merges.deinit(allocator);
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
        model.vocab_hash.deinit(allocator);
        model.merges.deinit(allocator);
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
        model.vocab_hash.deinit(allocator);
        model.merges.deinit(allocator);
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
        .merge_strings = .{},
        .byte_to_unicode = undefined,
        .unicode_to_byte = [_]i32{-1} ** 65536,
        .byte_fallback_ids = [_]i32{-1} ** 256,
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
        .merge_strings = .{},
        .byte_to_unicode = undefined,
        .unicode_to_byte = [_]i32{-1} ** 65536,
        .byte_fallback_ids = [_]i32{-1} ** 256,
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
