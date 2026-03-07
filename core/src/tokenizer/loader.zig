//! Tokenizer Loader
//!
//! Loads tokenizer configurations from JSON files or strings.
//! Supports HuggingFace tokenizer.json format with fast vocab/merges parsing.

const std = @import("std");
const schema = @import("schema.zig");
const bpe = @import("bpe.zig");
const json_utils = @import("json_utils.zig");
const wordpiece_model = @import("wordpiece.zig");
const unigram_model = @import("unigram.zig");
const utils = @import("utils.zig");
const ct = @import("c_types.zig");
const tok_fns = @import("pipeline.zig");
const log = @import("../log.zig");

const PATTERN_GPT2: [:0]const u8 = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
const PATTERN_BERT: [:0]const u8 = "[A-Za-z0-9]+|[^A-Za-z0-9\\s]+";
const PATTERN_WS: [:0]const u8 = "[^\\s]+";

const ManagedArrayList = std.array_list.Managed;

// -------------------- Streaming Loader (hot path) --------------------

/// Fast tokenizer loader - uses direct scanning for vocab/merges (the heavy parts)
/// Falls back to std.json only for small metadata sections
pub fn load_from_slice_streaming(allocator: std.mem.Allocator, json_content: []const u8) !schema.TokenizerRoot {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();

    const model_section = findSection(json_content, "\"model\"") orelse {
        log.warn("tokenizer", "Tokenizer model parse failed", .{
            .stage = "model_section_missing",
            .json_bytes = json_content.len,
        });
        return error.InvalidModel;
    };
    if (model_section.len == 0 or model_section[0] != '{') {
        log.warn("tokenizer", "Tokenizer model parse failed", .{
            .stage = "model_section_not_object",
            .json_bytes = json_content.len,
        });
        return error.InvalidModel;
    }
    const model_end = findMatchingBrace(model_section, '{', '}') orelse {
        log.warn("tokenizer", "Tokenizer model parse failed", .{
            .stage = "model_section_unbalanced",
            .json_bytes = json_content.len,
        });
        return error.InvalidModel;
    };
    const model_json = model_section[0..model_end];
    log.debug("tokenizer", "Tokenizer fast-parse stage", .{ .stage = "model_section", .model_bytes = model_json.len }, @src());

    // Fast path: directly scan for vocab and merges sections
    var vocab_entries = ManagedArrayList(schema.TokenId).init(arena_allocator);
    var merge_entries = ManagedArrayList([]const u8).init(arena_allocator);
    const model_type_name = findJsonFieldString(model_json, "\"type\"") orelse {
        log.warn("tokenizer", "Tokenizer model parse failed", .{
            .stage = "model_type_missing",
            .model_bytes = model_json.len,
        });
        return error.InvalidModel;
    };
    if (!std.mem.eql(u8, model_type_name, "BPE") and
        !std.mem.eql(u8, model_type_name, "WordPiece") and
        !std.mem.eql(u8, model_type_name, "Unigram"))
    {
        log.warn("tokenizer", "Tokenizer model parse failed", .{
            .stage = "model_type_unsupported",
            .model_type = model_type_name,
        });
        return error.UnsupportedModel;
    }
    var vocab_is_array = false;

    // Find and parse vocab section directly
    if (findSection(model_json, "\"vocab\"")) |vocab_section| {
        if (vocab_section.len > 0 and vocab_section[0] == '{') {
            // Find matching closing brace
            const vocab_end = findMatchingBrace(vocab_section, '{', '}') orelse vocab_section.len;
            try parseVocabFastSection(arena_allocator, vocab_section[0..vocab_end], &vocab_entries);
        } else if (vocab_section.len > 0 and vocab_section[0] == '[') {
            vocab_is_array = true;
            // Unigram array format: [["token", score], ...]
            // Use std.json Scanner for correctness (this path is rare)
            const vocab_end = findMatchingBrace(vocab_section, '[', ']') orelse vocab_section.len;
            const vocab_json = vocab_section[0 .. vocab_end + 1];
            var scanner = std.json.Scanner.initCompleteInput(arena_allocator, vocab_json);
            if ((try scanner.next()) == .array_begin) {
                var next_id: i32 = 0;
                while (true) {
                    const tok = try scanner.next();
                    if (tok == .array_end) break;
                    if (tok != .array_begin) return error.InvalidVocab;
                    // Inner array: ["token", score]
                    const token_tok = try scanner.nextAlloc(arena_allocator, .alloc_if_needed);
                    const token_str: []const u8 = switch (token_tok) {
                        .string => |s| s,
                        .allocated_string => |s| s,
                        else => return error.InvalidVocab,
                    };
                    if (token_str.len == 0 or std.mem.indexOfScalar(u8, token_str, 0) != null) return error.InvalidVocab;
                    const score_tok = try scanner.next();
                    const score: f32 = switch (score_tok) {
                        .number => |n| std.fmt.parseFloat(f32, n) catch return error.InvalidVocab,
                        else => return error.InvalidVocab,
                    };
                    if ((try scanner.next()) != .array_end) return error.InvalidVocab;
                    try vocab_entries.append(.{ .token = token_str, .id = next_id, .score = score });
                    next_id += 1;
                }
            } else {
                return error.InvalidVocab;
            }
        } else {
            return error.InvalidVocab;
        }
    } else {
        return error.InvalidVocab;
    }

    // Find and parse merges section directly
    if (findSection(model_json, "\"merges\"")) |merges_section| {
        if (merges_section.len > 0 and merges_section[0] == '[') {
            const merges_end = findMatchingBrace(merges_section, '[', ']') orelse merges_section.len;
            try parseMergesFastSection(arena_allocator, merges_section[0..merges_end], &merge_entries);
        } else {
            return error.InvalidMerges;
        }
    }
    if (std.mem.eql(u8, model_type_name, "WordPiece") and vocab_is_array) return error.InvalidVocab;
    if (std.mem.eql(u8, model_type_name, "Unigram") and !vocab_is_array) return error.InvalidVocab;
    if (std.mem.eql(u8, model_type_name, "BPE") and vocab_is_array) return error.InvalidVocab;
    log.debug("tokenizer", "Tokenizer fast-parse stage", .{
        .stage = "model_core_parsed",
        .model_type = model_type_name,
        .vocab_entries = vocab_entries.items.len,
        .merge_entries = merge_entries.items.len,
    }, @src());
    try validateModelOptions(model_type_name, model_json);
    try validateBpeMergeReferences(arena_allocator, model_type_name, vocab_entries.items, merge_entries.items);
    log.debug("tokenizer", "Tokenizer fast-parse stage", .{ .stage = "model_validated" }, @src());

    // Extract unk_token from model section
    var unk_token_str: ?[]const u8 = null;
    const search_len = @min(500, model_json.len);
    const search_region = model_json[0..search_len];
    // Try "unk_token" (string value) — used by WordPiece
    if (std.mem.indexOf(u8, search_region, "\"unk_token\"")) |unk_pos| {
        const after_key = model_json[unk_pos + "\"unk_token\"".len ..];
        if (findQuotedString(after_key)) |val| {
            unk_token_str = val;
        }
    }
    // Try "unk_id" (integer index into vocab) — used by Unigram
    if (unk_token_str == null) {
        if (std.mem.indexOf(u8, search_region, "\"unk_id\"")) |unk_pos| {
            const after_key = model_json[unk_pos + "\"unk_id\"".len ..];
            // Skip colon and whitespace to find the integer
            var skip: usize = 0;
            while (skip < after_key.len and (after_key[skip] == ':' or after_key[skip] == ' ' or after_key[skip] == '\t')) : (skip += 1) {}
            if (skip < after_key.len) {
                const num_start = skip;
                while (skip < after_key.len and after_key[skip] >= '0' and after_key[skip] <= '9') : (skip += 1) {}
                if (skip > num_start) {
                    const unk_id = std.fmt.parseInt(usize, after_key[num_start..skip], 10) catch null;
                    if (unk_id) |id| {
                        if (id < vocab_entries.items.len) {
                            unk_token_str = vocab_entries.items[id].token;
                        }
                    }
                }
            }
        }
    }

    // Extract max_input_chars_per_word from model section (used by WordPiece)
    var max_input_chars_per_word: i32 = 200;
    const mic_search_len = @min(2000, model_json.len);
    if (std.mem.indexOf(u8, model_json[0..mic_search_len], "\"max_input_chars_per_word\"")) |mic_pos| {
        const after_key = model_json[mic_pos + "\"max_input_chars_per_word\"".len ..];
        var skip: usize = 0;
        while (skip < after_key.len and (after_key[skip] == ':' or after_key[skip] == ' ' or after_key[skip] == '\t')) : (skip += 1) {}
        if (skip < after_key.len) {
            const num_start = skip;
            while (skip < after_key.len and after_key[skip] >= '0' and after_key[skip] <= '9') : (skip += 1) {}
            if (skip > num_start) {
                max_input_chars_per_word = std.fmt.parseInt(i32, after_key[num_start..skip], 10) catch 200;
            }
        }
    }

    // Parse small sections with std.json (added_tokens, normalizer, etc.)
    var added_token_entries = ManagedArrayList(schema.AddedToken).init(arena_allocator);
    var normalizer_spec: schema.Normalizer = .{};
    var pretokenizer_spec: schema.PreTokenizer = .{};
    var postprocessor_spec: schema.PostProcessor = .{};
    var decoder_spec: schema.Decoder = .{};

    try parseMetadataSections(
        arena_allocator,
        json_content,
        &added_token_entries,
        &normalizer_spec,
        &pretokenizer_spec,
        &postprocessor_spec,
        &decoder_spec,
    );
    log.debug("tokenizer", "Tokenizer fast-parse stage", .{
        .stage = "metadata_parsed",
        .added_tokens = added_token_entries.items.len,
    }, @src());

    var vocab_by_id = std.AutoHashMap(i32, []const u8).init(arena_allocator);
    defer vocab_by_id.deinit();
    for (vocab_entries.items) |entry| {
        try vocab_by_id.put(entry.id, entry.token);
    }
    for (added_token_entries.items) |entry| {
        const vocab_token = vocab_by_id.get(entry.id) orelse continue;
        if (!std.mem.eql(u8, vocab_token, entry.content)) return error.InvalidAdded;
    }

    log.trace("tokenizer", "Parsed vocab and merges", .{
        .vocab_entries = vocab_entries.items.len,
        .merge_entries = merge_entries.items.len,
    }, @src());

    return schema.TokenizerRoot{
        .version = null,
        .model = .{
            .type = model_type_name,
            .vocab = try vocab_entries.toOwnedSlice(),
            .merges = if (merge_entries.items.len > 0) try merge_entries.toOwnedSlice() else null,
            .unk_token = unk_token_str,
            .bos_token = null,
            .eos_token = null,
            .max_input_chars_per_word = max_input_chars_per_word,
        },
        .added_tokens = try added_token_entries.toOwnedSlice(),
        .normalizer = normalizer_spec,
        .pre_tokenizer = pretokenizer_spec,
        .post_processor = postprocessor_spec,
        .decoder = decoder_spec,
    };
}

fn parseMetadataSections(
    arena_allocator: std.mem.Allocator,
    json_content: []const u8,
    added_token_entries: *ManagedArrayList(schema.AddedToken),
    normalizer: *schema.Normalizer,
    pretokenizer: *schema.PreTokenizer,
    postprocessor: *schema.PostProcessor,
    decoder: *schema.Decoder,
) !void {
    var direct_bytefallback_decoder = false;
    if (findSection(json_content, "\"decoder\"")) |section| {
        if (section.len > 0 and section[0] == '{') {
            const end = findMatchingBrace(section, '{', '}') orelse return error.InvalidDecoder;
            const decoder_section = section[0..end];
            try ensureJsonDepthWithinLimit(decoder_section, error.InvalidDecoder);
            if (findJsonFieldString(decoder_section, "\"type\"")) |decoder_type| {
                direct_bytefallback_decoder =
                    std.mem.eql(u8, decoder_type, "ByteFallback") and
                    std.mem.indexOf(u8, decoder_section, "\"decoders\"") == null;
            }
        }
    }

    // Use std.json for the small metadata sections
    var scanner = std.json.Scanner.initCompleteInput(arena_allocator, json_content);
    if ((try scanner.next()) == .object_begin) {
        while (true) {
            const json_token = try scanner.nextAlloc(arena_allocator, .alloc_if_needed);
            switch (json_token) {
                .object_end => break,
                .string, .allocated_string => |field_name| {
                    if (std.mem.eql(u8, field_name, "added_tokens")) {
                        try parseAddedTokens(arena_allocator, &scanner, added_token_entries);
                    } else if (std.mem.eql(u8, field_name, "normalizer")) {
                        normalizer.* = try parseNormalizer(arena_allocator, &scanner);
                    } else if (std.mem.eql(u8, field_name, "pre_tokenizer")) {
                        pretokenizer.* = try parsePreTokenizer(arena_allocator, &scanner);
                    } else if (std.mem.eql(u8, field_name, "post_processor")) {
                        postprocessor.* = try parsePostProcessor(arena_allocator, &scanner);
                    } else if (std.mem.eql(u8, field_name, "decoder")) {
                        if (direct_bytefallback_decoder) {
                            try scanner.skipValue();
                            decoder.* = .{ .type = "ByteFallback" };
                        } else {
                            decoder.* = try parseDecoder(arena_allocator, &scanner);
                        }
                    } else {
                        try scanner.skipValue();
                    }
                },
                else => break,
            }
        }
    }
}

fn validateModelOptions(model_type_name: []const u8, model_json: []const u8) !void {
    if (!std.mem.eql(u8, model_type_name, "BPE")) return;

    // HuggingFace BPE configs often emit these fields with a neutral default
    // value even when the feature is effectively disabled. Accept the inert
    // form so real-world tokenizers still load, but continue rejecting any
    // non-default value that would silently change runtime behavior.
    if (findJsonFieldValue(model_json, "\"dropout\"")) |value| {
        if (!std.mem.eql(u8, value, "null")) {
            log.warn("tokenizer", "Tokenizer BPE option rejected", .{
                .field = "dropout",
                .value_len = value.len,
                .value_preview = value[0..@min(value.len, 32)],
            });
            return error.InvalidModel;
        }
    }
    if (findJsonFieldValue(model_json, "\"fuse_unk\"")) |value| {
        if (!std.mem.eql(u8, value, "false")) {
            log.warn("tokenizer", "Tokenizer BPE option rejected", .{
                .field = "fuse_unk",
                .value_len = value.len,
                .value_preview = value[0..@min(value.len, 32)],
            });
            return error.InvalidModel;
        }
    }
    if (findJsonFieldValue(model_json, "\"byte_fallback\"")) |value| {
        if (!std.mem.eql(u8, value, "false")) {
            log.warn("tokenizer", "Tokenizer BPE option rejected", .{
                .field = "byte_fallback",
                .value_len = value.len,
                .value_preview = value[0..@min(value.len, 32)],
            });
            return error.InvalidModel;
        }
    }
    if (findJsonFieldValue(model_json, "\"continuing_subword_prefix\"")) |value| {
        if (!std.mem.eql(u8, value, "null") and !std.mem.eql(u8, value, "\"\"")) {
            log.warn("tokenizer", "Tokenizer BPE option rejected", .{
                .field = "continuing_subword_prefix",
                .value_len = value.len,
                .value_preview = value[0..@min(value.len, 32)],
            });
            return error.InvalidModel;
        }
    }
    if (findJsonFieldValue(model_json, "\"end_of_word_suffix\"")) |value| {
        if (!std.mem.eql(u8, value, "null") and !std.mem.eql(u8, value, "\"\"")) {
            log.warn("tokenizer", "Tokenizer BPE option rejected", .{
                .field = "end_of_word_suffix",
                .value_len = value.len,
                .value_preview = value[0..@min(value.len, 32)],
            });
            return error.InvalidModel;
        }
    }
    if (findJsonFieldValue(model_json, "\"ignore_merges\"")) |value| {
        if (!std.mem.eql(u8, value, "false")) {
            log.warn("tokenizer", "Tokenizer BPE option rejected", .{
                .field = "ignore_merges",
                .value_len = value.len,
                .value_preview = value[0..@min(value.len, 32)],
            });
            return error.InvalidModel;
        }
    }
}

fn validateBpeMergeReferences(
    arena_allocator: std.mem.Allocator,
    model_type_name: []const u8,
    vocab_entries: []const schema.TokenId,
    merge_entries: []const []const u8,
) !void {
    if (!std.mem.eql(u8, model_type_name, "BPE")) return;

    var vocab_tokens = std.StringHashMap(void).init(arena_allocator);
    defer vocab_tokens.deinit();
    for (vocab_entries) |entry| {
        try vocab_tokens.put(entry.token, {});
    }

    for (merge_entries) |merge_entry| {
        const split_at = std.mem.indexOfScalar(u8, merge_entry, ' ') orelse return error.InvalidMerges;
        const lhs = merge_entry[0..split_at];
        const rhs = merge_entry[split_at + 1 ..];
        if (!vocab_tokens.contains(lhs) or !vocab_tokens.contains(rhs)) return error.InvalidMerges;

        // Later merge rules are allowed to reference intermediate merge outputs
        // even when those merged tokens are not declared in the original vocab.
        const merged = try std.mem.concat(arena_allocator, u8, &.{ lhs, rhs });
        try vocab_tokens.put(merged, {});
    }
}

// Use shared JSON parsing utilities
const findSection = utils.findJsonSection;
const findMatchingBrace = utils.findMatchingBrace;

/// Find first quoted string value
fn findQuotedString(s: []const u8) ?[]const u8 {
    var scan_idx: usize = 0;
    // Find opening quote
    while (scan_idx < s.len and s[scan_idx] != '"') : (scan_idx += 1) {}
    if (scan_idx >= s.len) return null;
    scan_idx += 1;
    const start = scan_idx;
    // Find closing quote
    while (scan_idx < s.len and s[scan_idx] != '"') {
        if (s[scan_idx] == '\\') scan_idx += 2 else scan_idx += 1;
    }
    return s[start..scan_idx];
}

fn skipJsonValueFast(json_bytes: []const u8, cursor_ptr: *usize) !void {
    var cursor = cursor_ptr.*;
    if (cursor >= json_bytes.len) return;

    switch (json_bytes[cursor]) {
        '{' => {
            const end = findMatchingBrace(json_bytes[cursor..], '{', '}') orelse return error.InvalidVocab;
            cursor += end;
        },
        '[' => {
            const end = findMatchingBrace(json_bytes[cursor..], '[', ']') orelse return error.InvalidVocab;
            cursor += end;
        },
        '"' => {
            cursor += 1;
            while (cursor < json_bytes.len and json_bytes[cursor] != '"') {
                if (json_bytes[cursor] == '\\') {
                    cursor += 2;
                } else {
                    cursor += 1;
                }
            }
            if (cursor >= json_bytes.len) return error.InvalidVocab;
            cursor += 1;
        },
        else => {
            while (cursor < json_bytes.len and json_bytes[cursor] != ',' and json_bytes[cursor] != '}') : (cursor += 1) {}
        },
    }

    cursor_ptr.* = cursor;
}

/// Fast merges parser - handles both ["a", "b"] and "a b" formats
fn parseMergesFastSection(arena_allocator: std.mem.Allocator, json_bytes: []const u8, merge_entries: *ManagedArrayList([]const u8)) !void {
    // Some models have 500k+ merges, so use a large capacity
    try merge_entries.ensureTotalCapacity(600000);
    var seen_merges = std.StringHashMap(void).init(arena_allocator);
    defer seen_merges.deinit();

    var cursor: usize = 0;
    while (cursor < json_bytes.len and std.ascii.isWhitespace(json_bytes[cursor])) : (cursor += 1) {}
    if (cursor < json_bytes.len and json_bytes[cursor] == '[') cursor += 1;
    while (cursor < json_bytes.len) {
        // Look for either [ (array format) or " (string format)
        while (cursor < json_bytes.len and json_bytes[cursor] != '[' and json_bytes[cursor] != '"') : (cursor += 1) {}
        if (cursor >= json_bytes.len) break;

        if (json_bytes[cursor] == '[') {
            // Array format: ["a", "b"]
            cursor += 1;
            while (cursor < json_bytes.len and std.ascii.isWhitespace(json_bytes[cursor])) : (cursor += 1) {}
            if (cursor < json_bytes.len and json_bytes[cursor] == ']') {
                cursor += 1;
                continue;
            }

            // Find first string
            if (cursor >= json_bytes.len or json_bytes[cursor] != '"') return error.InvalidMerges;
            cursor += 1;
            const a_start = cursor;
            var a_escape = false;
            while (cursor < json_bytes.len and json_bytes[cursor] != '"') {
                if (json_bytes[cursor] == '\\') {
                    a_escape = true;
                    cursor += 2;
                } else cursor += 1;
            }
            if (cursor >= json_bytes.len) break;
            const a_end = cursor;
            cursor += 1;
            while (cursor < json_bytes.len and (std.ascii.isWhitespace(json_bytes[cursor]) or json_bytes[cursor] == ',')) : (cursor += 1) {}

            // Find second string
            if (cursor >= json_bytes.len or json_bytes[cursor] != '"') return error.InvalidMerges;
            cursor += 1;
            const b_start = cursor;
            var b_escape = false;
            while (cursor < json_bytes.len and json_bytes[cursor] != '"') {
                if (json_bytes[cursor] == '\\') {
                    b_escape = true;
                    cursor += 2;
                } else cursor += 1;
            }
            if (cursor >= json_bytes.len) break;
            const b_end = cursor;
            cursor += 1;

            // Get strings
            const lhs = if (a_escape) try unescapeJsonStringFast(arena_allocator, json_bytes[a_start..a_end]) else json_bytes[a_start..a_end];
            const rhs = if (b_escape) try unescapeJsonStringFast(arena_allocator, json_bytes[b_start..b_end]) else json_bytes[b_start..b_end];
            if (lhs.len == 0 or rhs.len == 0 or std.mem.indexOfScalar(u8, lhs, 0) != null or std.mem.indexOfScalar(u8, rhs, 0) != null) {
                return error.InvalidMerges;
            }

            // Join with space
            const merged = try std.fmt.allocPrint(arena_allocator, "{s} {s}", .{ lhs, rhs });
            if (seen_merges.contains(merged)) return error.InvalidMerges;
            try seen_merges.put(merged, {});
            merge_entries.appendAssumeCapacity(merged);

            // Skip to end of array
            while (cursor < json_bytes.len and std.ascii.isWhitespace(json_bytes[cursor])) : (cursor += 1) {}
            if (cursor >= json_bytes.len or json_bytes[cursor] != ']') return error.InvalidMerges;
            cursor += 1;
        } else {
            // String format: "a b"
            cursor += 1;
            const start = cursor;
            var has_escape = false;
            while (cursor < json_bytes.len and json_bytes[cursor] != '"') {
                if (json_bytes[cursor] == '\\') {
                    has_escape = true;
                    cursor += 2;
                } else cursor += 1;
            }
            if (cursor >= json_bytes.len) break;

            const merge_str = if (has_escape) try unescapeJsonStringFast(arena_allocator, json_bytes[start..cursor]) else json_bytes[start..cursor];
            const first_space = std.mem.indexOfScalar(u8, merge_str, ' ') orelse return error.InvalidMerges;
            if (first_space == 0 or first_space + 1 >= merge_str.len) return error.InvalidMerges;
            if (std.mem.indexOfScalarPos(u8, merge_str, first_space + 1, ' ') != null) return error.InvalidMerges;
            if (seen_merges.contains(merge_str)) return error.InvalidMerges;
            try seen_merges.put(merge_str, {});
            merge_entries.appendAssumeCapacity(merge_str);
            cursor += 1;
        }

        while (cursor < json_bytes.len and (std.ascii.isWhitespace(json_bytes[cursor]) or json_bytes[cursor] == ',')) : (cursor += 1) {}
        if (cursor < json_bytes.len and json_bytes[cursor] == ']') break;
    }
}

fn parseModel(arena_allocator: std.mem.Allocator, json_scanner: *std.json.Scanner) !schema.Model {
    if ((try json_scanner.next()) != .object_begin) return error.InvalidModel;
    var model_type_name: []const u8 = "";
    var vocab_entries = ManagedArrayList(schema.TokenId).init(arena_allocator);
    var merge_entries = ManagedArrayList([]const u8).init(arena_allocator);
    var unk_token_bytes: ?[]const u8 = null;
    var bos_token_bytes: ?[]const u8 = null;
    var eos_token_bytes: ?[]const u8 = null;
    var is_unigram_vocab = false;
    var max_input_chars: i32 = 200;

    while (true) {
        const json_token = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
        switch (json_token) {
            .object_end => break,
            .allocated_string => |key| {
                if (std.mem.eql(u8, key, "type")) {
                    const type_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    if (type_value == .allocated_string) model_type_name = type_value.allocated_string;
                } else if (std.mem.eql(u8, key, "vocab")) {
                    try parseVocab(arena_allocator, json_scanner, &vocab_entries, &is_unigram_vocab);
                } else if (std.mem.eql(u8, key, "merges")) {
                    try parseMerges(arena_allocator, json_scanner, &merge_entries);
                } else if (std.mem.eql(u8, key, "unk_token")) {
                    const token_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    if (token_value == .allocated_string) unk_token_bytes = token_value.allocated_string;
                } else if (std.mem.eql(u8, key, "bos_token")) {
                    const token_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    if (token_value == .allocated_string) bos_token_bytes = token_value.allocated_string;
                } else if (std.mem.eql(u8, key, "eos_token")) {
                    const token_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    if (token_value == .allocated_string) eos_token_bytes = token_value.allocated_string;
                } else if (std.mem.eql(u8, key, "max_input_chars_per_word")) {
                    const val = try json_scanner.next();
                    if (val == .number) {
                        max_input_chars = std.fmt.parseInt(i32, val.number, 10) catch 200;
                    }
                } else {
                    // Skip unknown model fields (dropout, fuse_unk, byte_fallback, etc.)
                    try json_scanner.skipValue();
                }
            },
            else => return error.InvalidModel,
        }
    }

    return schema.Model{
        .type = model_type_name,
        .vocab = try vocab_entries.toOwnedSlice(),
        .merges = if (merge_entries.items.len > 0) try merge_entries.toOwnedSlice() else null,
        .unk_token = unk_token_bytes,
        .bos_token = bos_token_bytes,
        .eos_token = eos_token_bytes,
        .max_input_chars_per_word = max_input_chars,
    };
}

/// Fast direct vocab parser - bypasses std.json for speed
/// Scans for "key": number patterns directly in the JSON buffer
fn parseVocabFastSection(
    arena_allocator: std.mem.Allocator,
    json_bytes: []const u8,
    vocab_entries: *ManagedArrayList(schema.TokenId),
) !void {
    // Pre-allocate for large vocabularies (some models have 250k+ tokens)
    try vocab_entries.ensureTotalCapacity(300000);
    var seen_tokens = std.StringHashMap(void).init(arena_allocator);
    defer seen_tokens.deinit();
    var seen_ids = std.AutoHashMap(i32, void).init(arena_allocator);
    defer seen_ids.deinit();

    var cursor: usize = 0;
    while (cursor < json_bytes.len) {
        // Find opening quote for key
        while (cursor < json_bytes.len and json_bytes[cursor] != '"') : (cursor += 1) {}
        if (cursor >= json_bytes.len) break;
        cursor += 1; // skip opening quote

        // Find closing quote (handle escapes)
        const key_start = cursor;
        var has_escape = false;
        while (cursor < json_bytes.len and json_bytes[cursor] != '"') {
            if (json_bytes[cursor] == '\\') {
                has_escape = true;
                cursor += 2; // skip escape sequence
            } else {
                cursor += 1;
            }
        }
        if (cursor >= json_bytes.len) break;
        const key_end = cursor;
        cursor += 1; // skip closing quote

        // Skip whitespace and colon
        while (cursor < json_bytes.len and (json_bytes[cursor] == ' ' or json_bytes[cursor] == ':' or json_bytes[cursor] == '\t' or json_bytes[cursor] == '\n' or json_bytes[cursor] == '\r')) : (cursor += 1) {}

        // Check if value is a number (vocab entry) or something else (skip)
        if (cursor >= json_bytes.len) break;
        const ch = json_bytes[cursor];
        if (ch == '-') return error.InvalidVocab;
        if (ch >= '0' and ch <= '9') {
            // Parse number
            const num_start = cursor;
            while (cursor < json_bytes.len and json_bytes[cursor] >= '0' and json_bytes[cursor] <= '9') : (cursor += 1) {}
            if (cursor < json_bytes.len and (json_bytes[cursor] == '.' or json_bytes[cursor] == 'e' or json_bytes[cursor] == 'E')) {
                return error.InvalidVocab;
            }
            const id_num = std.fmt.parseInt(i32, json_bytes[num_start..cursor], 10) catch return error.InvalidVocab;

            // Get the key - zero-copy if no escapes
            const key = if (has_escape)
                try unescapeJsonStringFast(arena_allocator, json_bytes[key_start..key_end])
            else
                json_bytes[key_start..key_end];
            if (key.len == 0 or std.mem.indexOfScalar(u8, key, 0) != null) return error.InvalidVocab;
            if (seen_tokens.contains(key) or seen_ids.contains(id_num)) return error.InvalidVocab;
            try seen_tokens.put(key, {});
            try seen_ids.put(id_num, {});

            vocab_entries.appendAssumeCapacity(.{ .token = key, .id = id_num, .score = -1.0 });
        } else {
            try skipJsonValueFast(json_bytes, &cursor);
        }
    }
}

fn unescapeJsonStringFast(arena_allocator: std.mem.Allocator, input_bytes: []const u8) ![]const u8 {
    const unescaped = try json_utils.unescapeJsonString(arena_allocator, input_bytes);
    if (unescaped.ptr == input_bytes.ptr and unescaped.len == input_bytes.len) {
        return try arena_allocator.dupe(u8, input_bytes);
    }
    return unescaped;
}

fn parseVocab(
    arena_allocator: std.mem.Allocator,
    json_scanner: *std.json.Scanner,
    vocab_entries: *ManagedArrayList(schema.TokenId),
    is_unigram: *bool,
) !void {
    const next_token = try json_scanner.next();
    switch (next_token) {
        .object_begin => {
            is_unigram.* = false;
            while (true) {
                const json_token = try json_scanner.nextAlloc(arena_allocator, .alloc_if_needed);
                switch (json_token) {
                    .object_end => break,
                    .string => |key| {
                        // Zero-copy: string has no escapes, points directly into JSON buffer
                        const value_token = try json_scanner.next();
                        const id_num: i32 = switch (value_token) {
                            .number => |bytes| std.fmt.parseInt(i32, bytes, 10) catch return error.InvalidVocab,
                            else => return error.InvalidVocab,
                        };
                        try vocab_entries.append(.{ .token = key, .id = id_num, .score = -1.0 });
                    },
                    .allocated_string => |key| {
                        // String had escapes, was allocated
                        const value_token = try json_scanner.next();
                        const id_num: i32 = switch (value_token) {
                            .number => |bytes| std.fmt.parseInt(i32, bytes, 10) catch return error.InvalidVocab,
                            else => return error.InvalidVocab,
                        };
                        try vocab_entries.append(.{ .token = key, .id = id_num, .score = -1.0 });
                    },
                    else => return error.InvalidVocab,
                }
            }
        },
        .array_begin => {
            is_unigram.* = true;
            var token_index: i32 = 0;
            while (true) {
                const json_token = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                switch (json_token) {
                    .array_end => break,
                    .array_begin => {
                        const token_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                        const token_slice = switch (token_value) {
                            .allocated_string => |s| s,
                            else => return error.InvalidVocab,
                        };
                        const score_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                        const score_val: f32 = switch (score_value) {
                            .allocated_number => |bytes| std.fmt.parseFloat(f32, bytes) catch return error.InvalidVocab,
                            .number => |bytes| std.fmt.parseFloat(f32, bytes) catch return error.InvalidVocab,
                            else => return error.InvalidVocab,
                        };
                        try vocab_entries.append(.{ .token = token_slice, .id = token_index, .score = score_val });
                        token_index += 1;
                        const closer = try json_scanner.next();
                        if (closer != .array_end) return error.InvalidVocab;
                    },
                    else => return error.InvalidVocab,
                }
            }
        },
        else => return error.InvalidVocab,
    }
}

fn parseMerges(arena_allocator: std.mem.Allocator, json_scanner: *std.json.Scanner, merge_entries: *ManagedArrayList([]const u8)) !void {
    if ((try json_scanner.next()) != .array_begin) return error.InvalidMerges;
    while (true) {
        const json_token = try json_scanner.nextAlloc(arena_allocator, .alloc_if_needed);
        switch (json_token) {
            .array_end => break,
            // String format: "a b" - zero-copy if no escapes
            .string => |s| try merge_entries.append(s),
            .allocated_string => |s| try merge_entries.append(s),
            // Array format: ["a", "b"] - join with space
            .array_begin => {
                const first_value = try json_scanner.nextAlloc(arena_allocator, .alloc_if_needed);
                const lhs = switch (first_value) {
                    .string => |s| s,
                    .allocated_string => |s| s,
                    else => return error.InvalidMerges,
                };
                const second_value = try json_scanner.nextAlloc(arena_allocator, .alloc_if_needed);
                const rhs = switch (second_value) {
                    .string => |s| s,
                    .allocated_string => |s| s,
                    else => return error.InvalidMerges,
                };
                // Expect array_end
                if ((try json_scanner.next()) != .array_end) return error.InvalidMerges;
                // Join "a" + " " + "b"
                const merged = try std.fmt.allocPrint(arena_allocator, "{s} {s}", .{ lhs, rhs });
                try merge_entries.append(merged);
            },
            else => return error.InvalidMerges,
        }
    }
}

fn parseAddedTokens(arena_allocator: std.mem.Allocator, json_scanner: *std.json.Scanner, added_entries: *ManagedArrayList(schema.AddedToken)) !void {
    if ((try json_scanner.next()) != .array_begin) return error.InvalidAdded;
    var seen_ids = std.AutoHashMap(i32, void).init(arena_allocator);
    defer seen_ids.deinit();
    var seen_content = std.StringHashMap(i32).init(arena_allocator);
    defer seen_content.deinit();
    while (true) {
        const json_token = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
        switch (json_token) {
            .array_end => break,
            .object_begin => {
                var added_entry = schema.AddedToken{
                    .id = 0,
                    .content = "",
                };
                var saw_id = false;
                var saw_content = false;
                while (true) {
                    const key_token = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    switch (key_token) {
                        .object_end => break,
                        .allocated_string => |key| {
                            if (std.mem.eql(u8, key, "id")) {
                                const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                                added_entry.id = switch (value) {
                                    .allocated_number => |bytes| std.fmt.parseInt(i32, bytes, 10) catch return error.InvalidAdded,
                                    .number => |bytes| std.fmt.parseInt(i32, bytes, 10) catch return error.InvalidAdded,
                                    else => return error.InvalidAdded,
                                };
                                if (added_entry.id < 0) return error.InvalidAdded;
                                saw_id = true;
                            } else if (std.mem.eql(u8, key, "content")) {
                                const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                                added_entry.content = switch (value) {
                                    .allocated_string => |s| s,
                                    .string => |s| s,
                                    else => return error.InvalidAdded,
                                };
                                saw_content = true;
                            } else if (std.mem.eql(u8, key, "single_word")) {
                                const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                                added_entry.single_word = (value == .true);
                            } else if (std.mem.eql(u8, key, "lstrip")) {
                                const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                                added_entry.lstrip = (value == .true);
                            } else if (std.mem.eql(u8, key, "rstrip")) {
                                const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                                added_entry.rstrip = (value == .true);
                            } else if (std.mem.eql(u8, key, "normalized")) {
                                const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                                added_entry.normalized = (value == .true);
                            } else if (std.mem.eql(u8, key, "special")) {
                                const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                                added_entry.special = (value == .true);
                            } else {
                                // Skip unknown added_token fields
                                try json_scanner.skipValue();
                            }
                        },
                        else => return error.InvalidAdded,
                    }
                }
                if (!saw_id or !saw_content) return error.InvalidAdded;
                if (std.mem.indexOfScalar(u8, added_entry.content, 0) != null) return error.InvalidAdded;
                if (added_entry.content.len == 0) continue;
                if (seen_ids.contains(added_entry.id)) return error.InvalidAdded;
                if (seen_content.get(added_entry.content)) |existing_id| {
                    if (existing_id != added_entry.id) return error.InvalidAdded;
                } else {
                    try seen_content.put(added_entry.content, added_entry.id);
                }
                try seen_ids.put(added_entry.id, {});
                try added_entries.append(added_entry);
            },
            else => return error.InvalidAdded,
        }
    }
}

const MAX_JSON_PIPELINE_DEPTH: usize = 128;

fn ensureJsonDepthWithinLimit(json_bytes: []const u8, comptime err: anyerror) !void {
    // Match the recursive parser's actual risk surface: nested objects.
    // Arrays are containers but do not add parser recursion by themselves.
    // The recursive functions start at depth=0 for the outer object, so the
    // maximum object nesting permitted here is MAX_JSON_PIPELINE_DEPTH + 1.
    const max_object_depth = MAX_JSON_PIPELINE_DEPTH + 1;
    var object_depth: usize = 0;
    var cursor: usize = 0;
    while (cursor < json_bytes.len) {
        const byte = json_bytes[cursor];
        if (byte == '"') {
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
            continue;
        }
        if (byte == '{') {
            object_depth += 1;
            if (object_depth > max_object_depth) return err;
        } else if (byte == '}') {
            if (object_depth == 0) return err;
            object_depth -= 1;
        }
        cursor += 1;
    }
    if (object_depth != 0) return err;
}

fn parseNormalizer(arena_allocator: std.mem.Allocator, json_scanner: *std.json.Scanner) !schema.Normalizer {
    return parseNormalizerDepth(arena_allocator, json_scanner, 0);
}

fn parseNormalizerDepth(arena_allocator: std.mem.Allocator, json_scanner: *std.json.Scanner, depth: usize) !schema.Normalizer {
    if (depth > MAX_JSON_PIPELINE_DEPTH) return error.InvalidNormalizer;
    var normalizer = schema.Normalizer{};
    var saw_lowercase = false;
    var saw_strip_accents = false;
    var saw_clean_text = false;
    var saw_handle_chinese_chars = false;
    var saw_prepend = false;
    var saw_replace_pattern = false;
    const first_token = try json_scanner.next();
    if (first_token == .null) return normalizer;
    if (first_token != .object_begin) {
        log.warn("tokenizer", "Normalizer parse rejected non-object token", .{
            .token = @tagName(first_token),
        });
        return error.InvalidNormalizer;
    }

    while (true) {
        const json_token = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
        switch (json_token) {
            .object_end => break,
            .allocated_string => |key| {
                if (std.mem.eql(u8, key, "type")) {
                    const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    normalizer.type = switch (value) {
                        .allocated_string => |s| s,
                        .string => |s| s,
                        else => return error.InvalidNormalizer,
                    };
                } else if (std.mem.eql(u8, key, "lowercase")) {
                    const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    normalizer.lowercase = (value == .true);
                    saw_lowercase = true;
                } else if (std.mem.eql(u8, key, "strip_accents")) {
                    const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    normalizer.strip_accents = (value == .true);
                    saw_strip_accents = true;
                } else if (std.mem.eql(u8, key, "nfc") or std.mem.eql(u8, key, "NFC")) {
                    const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    normalizer.nfc = (value == .true);
                } else if (std.mem.eql(u8, key, "nfd") or std.mem.eql(u8, key, "NFD")) {
                    const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    normalizer.nfd = (value == .true);
                } else if (std.mem.eql(u8, key, "nfkc") or std.mem.eql(u8, key, "NFKC")) {
                    const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    normalizer.nfkc = (value == .true);
                } else if (std.mem.eql(u8, key, "nfkd") or std.mem.eql(u8, key, "NFKD")) {
                    const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    normalizer.nfkd = (value == .true);
                } else if (std.mem.eql(u8, key, "clean_text")) {
                    const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    normalizer.clean_text = (value == .true);
                    saw_clean_text = true;
                } else if (std.mem.eql(u8, key, "handle_chinese_chars")) {
                    const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    normalizer.handle_chinese_chars = (value == .true);
                    saw_handle_chinese_chars = true;
                } else if (std.mem.eql(u8, key, "prepend")) {
                    // Prepend normalizer: prepend this string to input
                    const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    normalizer.prepend = switch (value) {
                        .allocated_string => |s| s,
                        .string => |s| s,
                        else => return error.InvalidNormalizer,
                    };
                    saw_prepend = true;
                } else if (std.mem.eql(u8, key, "pattern")) {
                    // Replace normalizer pattern - can be {"String": "..."} or just a string
                    const pattern_token = try json_scanner.next();
                    if (pattern_token == .object_begin) {
                        // Parse {"String": "..."} or {"Regex": "..."}
                        while (true) {
                            const pattern_entry = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                            switch (pattern_entry) {
                                .object_end => break,
                                .allocated_string => |pat_key| {
                                    if (std.mem.eql(u8, pat_key, "String") or std.mem.eql(u8, pat_key, "Regex")) {
                                        const pat_val = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                                        normalizer.replace_pattern = switch (pat_val) {
                                            .allocated_string => |s| s,
                                            .string => |s| s,
                                            else => return error.InvalidNormalizer,
                                        };
                                        saw_replace_pattern = true;
                                    } else {
                                        try json_scanner.skipValue();
                                    }
                                },
                                else => return error.InvalidNormalizer,
                            }
                        }
                    } else switch (pattern_token) {
                        .allocated_string => |s| {
                            normalizer.replace_pattern = s;
                            saw_replace_pattern = true;
                        },
                        .string => |s| {
                            normalizer.replace_pattern = s;
                            saw_replace_pattern = true;
                        },
                        else => return error.InvalidNormalizer,
                    }
                } else if (std.mem.eql(u8, key, "content")) {
                    // Replace normalizer content (replacement string)
                    const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    normalizer.replace_content = switch (value) {
                        .allocated_string => |s| s,
                        .string => |s| s,
                        else => return error.InvalidNormalizer,
                    };
                } else if (std.mem.eql(u8, key, "normalizers")) {
                    // Sequence normalizer - parse the array and aggregate settings
                    if ((try json_scanner.next()) != .array_begin) return error.InvalidNormalizer;
                    while (true) {
                        const arr_token = try json_scanner.peekNextTokenType();
                        if (arr_token == .array_end) {
                            _ = try json_scanner.next();
                            break;
                        }
                        // Recursively parse each sub-normalizer
                        const sub = try parseNormalizerDepth(arena_allocator, json_scanner, depth + 1);
                        // Aggregate: OR the boolean flags
                        normalizer.lowercase = normalizer.lowercase or sub.lowercase;
                        normalizer.strip_accents = normalizer.strip_accents or sub.strip_accents;
                        normalizer.nfc = normalizer.nfc or sub.nfc;
                        normalizer.nfd = normalizer.nfd or sub.nfd;
                        normalizer.nfkc = normalizer.nfkc or sub.nfkc;
                        normalizer.clean_text = normalizer.clean_text or sub.clean_text;
                        normalizer.handle_chinese_chars = normalizer.handle_chinese_chars or sub.handle_chinese_chars;
                        // Aggregate Prepend/Replace (take first non-null)
                        if (normalizer.prepend == null) normalizer.prepend = sub.prepend;
                        if (normalizer.replace_pattern == null) normalizer.replace_pattern = sub.replace_pattern;
                        if (normalizer.replace_content == null) normalizer.replace_content = sub.replace_content;
                    }
                } else {
                    try json_scanner.skipValue();
                }
            },
            else => return error.InvalidNormalizer,
        }
    }
    // Infer settings from type if not explicitly set
    if (std.mem.eql(u8, normalizer.type, "BertNormalizer")) {
        if (!saw_clean_text) normalizer.clean_text = true;
        if (!saw_handle_chinese_chars) normalizer.handle_chinese_chars = true;
        if (!saw_lowercase) normalizer.lowercase = true;
        if (!saw_strip_accents) normalizer.strip_accents = true;
    } else if (std.mem.eql(u8, normalizer.type, "Lowercase")) {
        normalizer.lowercase = true;
    } else if (std.mem.eql(u8, normalizer.type, "NFC")) {
        normalizer.nfc = true;
    } else if (std.mem.eql(u8, normalizer.type, "NFD")) {
        normalizer.nfd = true;
    } else if (std.mem.eql(u8, normalizer.type, "NFKC")) {
        normalizer.nfkc = true;
    } else if (std.mem.eql(u8, normalizer.type, "NFKD")) {
        normalizer.nfkd = true;
    } else if (std.mem.eql(u8, normalizer.type, "StripAccents")) {
        normalizer.strip_accents = true;
    } else if (std.mem.eql(u8, normalizer.type, "Prepend")) {
        if (!saw_prepend or normalizer.prepend == null) return error.InvalidNormalizer;
    } else if (std.mem.eql(u8, normalizer.type, "Replace")) {
        if (!saw_replace_pattern or normalizer.replace_pattern == null) return error.InvalidNormalizer;
    } else if (std.mem.eql(u8, normalizer.type, "Sequence")) {
        // already validated by recursive parsing
    } else if (std.mem.eql(u8, normalizer.type, "Custom")) {
        // Internal parser tests use Custom as a container for explicit flags.
    } else if (normalizer.type.len > 0) {
        return error.InvalidNormalizer;
    }
    return normalizer;
}

fn parsePreTokenizer(arena_allocator: std.mem.Allocator, json_scanner: *std.json.Scanner) !schema.PreTokenizer {
    return parsePreTokenizerDepth(arena_allocator, json_scanner, 0);
}

fn parsePreTokenizerDepth(arena_allocator: std.mem.Allocator, json_scanner: *std.json.Scanner, depth: usize) !schema.PreTokenizer {
    if (depth > MAX_JSON_PIPELINE_DEPTH) return error.InvalidPreTokenizer;
    var pretokenizer = schema.PreTokenizer{};
    var split_behavior: ?[]const u8 = null;
    const first = try json_scanner.next();
    if (first == .null) return pretokenizer;
    if (first != .object_begin) return error.InvalidPreTokenizer;
    var saw_type = false;
    var saw_pattern = false;
    var saw_sequence = false;

    while (true) {
        const json_token = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
        switch (json_token) {
            .object_end => break,
            .allocated_string => |key| {
                if (std.mem.eql(u8, key, "type")) {
                    const type_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    pretokenizer.type = switch (type_value) {
                        .allocated_string => |s| s,
                        .string => |s| s,
                        else => return error.InvalidPreTokenizer,
                    };
                    saw_type = true;
                } else if (std.mem.eql(u8, key, "behavior")) {
                    const behavior_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    split_behavior = switch (behavior_value) {
                        .allocated_string => |s| s,
                        .string => |s| s,
                        else => return error.InvalidPreTokenizer,
                    };
                } else if (std.mem.eql(u8, key, "add_prefix_space")) {
                    const flag_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    pretokenizer.add_prefix_space = (flag_value == .true);
                } else if (std.mem.eql(u8, key, "prepend_scheme")) {
                    const scheme_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    if (scheme_value == .allocated_string) {
                        const scheme = scheme_value.allocated_string;
                        if (std.mem.eql(u8, scheme, "first") or std.mem.eql(u8, scheme, "always")) {
                            pretokenizer.add_prefix_space = true;
                        } else if (std.mem.eql(u8, scheme, "never")) {
                            pretokenizer.add_prefix_space = false;
                        }
                    } else if (scheme_value == .string) {
                        const scheme = scheme_value.string;
                        if (std.mem.eql(u8, scheme, "first") or std.mem.eql(u8, scheme, "always")) {
                            pretokenizer.add_prefix_space = true;
                        } else if (std.mem.eql(u8, scheme, "never")) {
                            pretokenizer.add_prefix_space = false;
                        }
                    }
                } else if (std.mem.eql(u8, key, "trim_offsets")) {
                    const flag_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    pretokenizer.trim_offsets = (flag_value == .true);
                } else if (std.mem.eql(u8, key, "invert")) {
                    const flag_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    pretokenizer.regex_invert = (flag_value == .true);
                } else if (std.mem.eql(u8, key, "use_regex")) {
                    const flag_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    pretokenizer.use_regex = (flag_value == .true);
                } else if (std.mem.eql(u8, key, "pattern")) {
                    // Pattern can be a string or an object with "Regex" or "String" field
                    const pattern_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    if (pattern_value == .allocated_string) {
                        pretokenizer.pattern = pattern_value.allocated_string;
                        saw_pattern = true;
                    } else if (pattern_value == .string) {
                        pretokenizer.pattern = pattern_value.string;
                        saw_pattern = true;
                    } else if (pattern_value == .object_begin) {
                        // Parse {"Regex": "..."} or {"String": "..."} format
                        while (true) {
                            const pattern_token = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                            switch (pattern_token) {
                                .object_end => break,
                                .allocated_string => |pattern_key| {
                                    if (std.mem.eql(u8, pattern_key, "Regex") or std.mem.eql(u8, pattern_key, "String")) {
                                        const pattern_field = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                                        pretokenizer.pattern = switch (pattern_field) {
                                            .allocated_string => |s| s,
                                            .string => |s| s,
                                            else => return error.InvalidPreTokenizer,
                                        };
                                        saw_pattern = true;
                                    } else {
                                        try json_scanner.skipValue();
                                    }
                                },
                                else => return error.InvalidPreTokenizer,
                            }
                        }
                    }
                } else if (std.mem.eql(u8, key, "pretokenizers")) {
                    // Sequence pre_tokenizer
                    if ((try json_scanner.next()) != .array_begin) return error.InvalidPreTokenizer;
                    saw_sequence = true;
                    while (true) {
                        const arr_token = try json_scanner.peekNextTokenType();
                        if (arr_token == .array_end) {
                            _ = try json_scanner.next();
                            break;
                        }
                        const sub = try parsePreTokenizerDepth(arena_allocator, json_scanner, depth + 1);
                        // Aggregate settings
                        pretokenizer.add_prefix_space = pretokenizer.add_prefix_space or sub.add_prefix_space;
                        pretokenizer.byte_level = pretokenizer.byte_level or sub.byte_level;
                        pretokenizer.whitespace = pretokenizer.whitespace or sub.whitespace;
                        pretokenizer.punctuation = pretokenizer.punctuation or sub.punctuation;
                        // Take first non-null pattern
                        if (pretokenizer.pattern == null and sub.pattern != null) {
                            pretokenizer.pattern = sub.pattern;
                            pretokenizer.regex_split = sub.regex_split;
                            pretokenizer.regex_invert = sub.regex_invert;
                        }
                    }
                } else {
                    try json_scanner.skipValue();
                }
            },
            else => return error.InvalidPreTokenizer,
        }
    }
    if (!saw_type) return error.InvalidPreTokenizer;
    // Infer settings from type
    if (std.mem.eql(u8, pretokenizer.type, "ByteLevel")) {
        pretokenizer.byte_level = true;
    } else if (std.mem.eql(u8, pretokenizer.type, "Whitespace")) {
        pretokenizer.whitespace = true;
    } else if (std.mem.eql(u8, pretokenizer.type, "WhitespaceSplit")) {
        pretokenizer.whitespace = true;
    } else if (std.mem.eql(u8, pretokenizer.type, "Punctuation")) {
        pretokenizer.punctuation = true;
    } else if (std.mem.eql(u8, pretokenizer.type, "BertPreTokenizer")) {
        pretokenizer.whitespace = true;
        pretokenizer.punctuation = true;
    } else if (std.mem.eql(u8, pretokenizer.type, "Metaspace")) {
        pretokenizer.metaspace = true;
        pretokenizer.whitespace = true; // Split on spaces before ▁ replacement
    } else if (std.mem.eql(u8, pretokenizer.type, "Split")) {
        if (!saw_pattern or pretokenizer.pattern == null) return error.InvalidPreTokenizer;
        // For Split type, behavior determines whether we emit matches or split on pattern
        // "Isolated" = emit matches (regex_split = false)
        // Other behaviors = split on pattern (regex_split = true)
        if (split_behavior) |b| {
            pretokenizer.regex_split = !std.mem.eql(u8, b, "Isolated");
        } else {
            pretokenizer.regex_split = true; // Default: split on pattern
        }
    } else if (std.mem.eql(u8, pretokenizer.type, "Sequence")) {
        if (!saw_sequence) return error.InvalidPreTokenizer;
    } else {
        return error.InvalidPreTokenizer;
    }
    if (pretokenizer.regex_invert) {
        pretokenizer.regex_split = false;
    }
    return pretokenizer;
}

test "parsePreTokenizer captures invert" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{
        \\  "type": "Split",
        \\  "behavior": "Removed",
        \\  "invert": true,
        \\  "pattern": {"Regex": "\\s+"}
        \\}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const parsed = try parsePreTokenizer(arena.allocator(), &scanner);
    try std.testing.expect(parsed.regex_invert);
    try std.testing.expect(!parsed.regex_split);
}

const ParsedTemplateSequence = struct {
    has_sequence: bool = false,
    first_before_sequence: ?[]const u8 = null,
    first_after_sequence: ?[]const u8 = null,
};

fn parseTemplateSequence(
    arena_allocator: std.mem.Allocator,
    json_scanner: *std.json.Scanner,
    referenced_special_tokens: *std.StringHashMap(void),
    template_info: ?*ParsedTemplateSequence,
) !void {
    while (true) {
        const entry = try json_scanner.next();
        if (entry == .array_end) break;
        if (entry != .object_begin) return error.InvalidPostProcessor;
        while (true) {
            const field = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
            switch (field) {
                .object_end => break,
                .allocated_string => |field_name| {
                    if (std.mem.eql(u8, field_name, "SpecialToken")) {
                        if ((try json_scanner.next()) != .object_begin) return error.InvalidPostProcessor;
                        var saw_id = false;
                        while (true) {
                            const special_field = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                            switch (special_field) {
                                .object_end => break,
                                .allocated_string => |special_name| {
                                    if (std.mem.eql(u8, special_name, "id")) {
                                        const id_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                                        const id_str = switch (id_value) {
                                            .allocated_string => |s| s,
                                            .string => |s| s,
                                            else => return error.InvalidPostProcessor,
                                        };
                                        try referenced_special_tokens.put(id_str, {});
                                        if (template_info) |info| {
                                            if (!info.has_sequence) {
                                                if (info.first_before_sequence == null) info.first_before_sequence = id_str;
                                            } else if (info.first_after_sequence == null) {
                                                info.first_after_sequence = id_str;
                                            }
                                        }
                                        saw_id = true;
                                    } else {
                                        try json_scanner.skipValue();
                                    }
                                },
                                else => return error.InvalidPostProcessor,
                            }
                        }
                        if (!saw_id) return error.InvalidPostProcessor;
                    } else if (std.mem.eql(u8, field_name, "Sequence")) {
                        if (template_info) |info| info.has_sequence = true;
                        try json_scanner.skipValue();
                    } else {
                        try json_scanner.skipValue();
                    }
                },
                else => return error.InvalidPostProcessor,
            }
        }
    }
}

fn parsePostProcessor(arena_allocator: std.mem.Allocator, json_scanner: *std.json.Scanner) !schema.PostProcessor {
    return parsePostProcessorDepth(arena_allocator, json_scanner, 0);
}

fn parsePostProcessorDepth(arena_allocator: std.mem.Allocator, json_scanner: *std.json.Scanner, depth: usize) !schema.PostProcessor {
    if (depth > MAX_JSON_PIPELINE_DEPTH) return error.InvalidPostProcessor;
    var postprocessor = schema.PostProcessor{};
    const first = try json_scanner.next();
    if (first == .null) return postprocessor;
    if (first != .object_begin) return error.InvalidPostProcessor;
    var declared_type: []const u8 = "";
    var saw_type = false;
    var saw_single = false;
    var saw_special_tokens = false;
    var saw_processors = false;
    var single_template = ParsedTemplateSequence{};
    var referenced_special_tokens = std.StringHashMap(void).init(arena_allocator);
    defer referenced_special_tokens.deinit();
    var defined_special_tokens = std.StringHashMap(void).init(arena_allocator);
    defer defined_special_tokens.deinit();

    while (true) {
        const json_token = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
        switch (json_token) {
            .object_end => break,
            .allocated_string => |key| {
                if (std.mem.eql(u8, key, "type")) {
                    const type_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    declared_type = switch (type_value) {
                        .allocated_string => |s| s,
                        .string => |s| s,
                        else => return error.InvalidPostProcessor,
                    };
                    postprocessor.type = declared_type;
                    saw_type = true;
                } else if (std.mem.eql(u8, key, "cls")) {
                    // Parse [token, type_id] array
                    if ((try json_scanner.next()) != .array_begin) return error.InvalidPostProcessor;
                    const cls_tok = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    postprocessor.cls_token = switch (cls_tok) {
                        .allocated_string => |s| s,
                        .string => |s| s,
                        else => return error.InvalidPostProcessor,
                    };
                    try json_scanner.skipValue(); // skip type_id
                    if ((try json_scanner.next()) != .array_end) return error.InvalidPostProcessor;
                } else if (std.mem.eql(u8, key, "sep")) {
                    if ((try json_scanner.next()) != .array_begin) return error.InvalidPostProcessor;
                    const sep_tok = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    postprocessor.sep_token = switch (sep_tok) {
                        .allocated_string => |s| s,
                        .string => |s| s,
                        else => return error.InvalidPostProcessor,
                    };
                    try json_scanner.skipValue(); // skip type_id
                    if ((try json_scanner.next()) != .array_end) return error.InvalidPostProcessor;
                } else if (std.mem.eql(u8, key, "add_special_tokens")) {
                    const flag_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    postprocessor.add_special = (flag_value == .true);
                } else if (std.mem.eql(u8, key, "single")) {
                    if ((try json_scanner.next()) != .array_begin) return error.InvalidPostProcessor;
                    saw_single = true;
                    try parseTemplateSequence(arena_allocator, json_scanner, &referenced_special_tokens, &single_template);
                } else if (std.mem.eql(u8, key, "pair")) {
                    if ((try json_scanner.next()) != .array_begin) return error.InvalidPostProcessor;
                    try parseTemplateSequence(arena_allocator, json_scanner, &referenced_special_tokens, null);
                } else if (std.mem.eql(u8, key, "processors")) {
                    if ((try json_scanner.next()) != .array_begin) return error.InvalidPostProcessor;
                    saw_processors = true;
                    while (true) {
                        const next_token = try json_scanner.peekNextTokenType();
                        if (next_token == .array_end) {
                            _ = try json_scanner.next();
                            break;
                        }
                        const sub = try parsePostProcessorDepth(arena_allocator, json_scanner, depth + 1);
                        mergeParsedPostProcessor(&postprocessor, sub);
                    }
                } else if (std.mem.eql(u8, key, "special_tokens")) {
                    if ((try json_scanner.next()) != .object_begin) return error.InvalidPostProcessor;
                    saw_special_tokens = true;
                    while (true) {
                        const entry = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                        switch (entry) {
                            .object_end => break,
                            .allocated_string => |name| {
                                if ((try json_scanner.next()) != .object_begin) return error.InvalidPostProcessor;
                                var ids_valid = false;
                                while (true) {
                                    const field = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                                    switch (field) {
                                        .object_end => break,
                                        .allocated_string => |field_name| {
                                            if (std.mem.eql(u8, field_name, "ids")) {
                                                if ((try json_scanner.next()) != .array_begin) return error.InvalidPostProcessor;
                                                if ((try json_scanner.next()) != .number) return error.InvalidPostProcessor;
                                                if ((try json_scanner.next()) != .array_end) return error.InvalidPostProcessor;
                                                ids_valid = true;
                                            } else {
                                                try json_scanner.skipValue();
                                            }
                                        },
                                        else => return error.InvalidPostProcessor,
                                    }
                                }
                                if (!ids_valid) return error.InvalidPostProcessor;
                                try defined_special_tokens.put(name, {});
                            },
                            else => return error.InvalidPostProcessor,
                        }
                    }
                } else {
                    try json_scanner.skipValue();
                }
            },
            else => return error.InvalidPostProcessor,
        }
    }
    if (!saw_type) return error.InvalidPostProcessor;
    // Infer settings from type
    if (std.mem.eql(u8, declared_type, "BertProcessing")) {
        postprocessor.add_special = true;
        if (postprocessor.cls_token == null) postprocessor.cls_token = "[CLS]";
        if (postprocessor.sep_token == null) postprocessor.sep_token = "[SEP]";
    } else if (std.mem.eql(u8, declared_type, "TemplateProcessing")) {
        postprocessor.add_special = true;
        if (single_template.first_before_sequence) |cls| postprocessor.cls_token = cls;
        if (single_template.first_after_sequence) |sep| postprocessor.sep_token = sep;
    } else if (std.mem.eql(u8, declared_type, "RobertaProcessing")) {
        postprocessor.add_special = true;
        postprocessor.pair = true; // RoBERTa uses double SEP in pair encoding
        if (postprocessor.cls_token == null) postprocessor.cls_token = "<s>";
        if (postprocessor.sep_token == null) postprocessor.sep_token = "</s>";
    } else if (std.mem.eql(u8, declared_type, "Sequence")) {
        if (!saw_processors) return error.InvalidPostProcessor;
    } else if (std.mem.eql(u8, declared_type, "ByteLevel")) {
        // ByteLevel post-processors only affect offsets; keep them as a valid no-op
        // in the strict loader so Sequence(ByteLevel, TemplateProcessing) is accepted.
    } else {
        return error.InvalidPostProcessor;
    }
    if (std.mem.eql(u8, declared_type, "TemplateProcessing") and (!saw_single or !saw_special_tokens)) {
        return error.InvalidPostProcessor;
    }
    if (std.mem.eql(u8, declared_type, "TemplateProcessing")) {
        var referenced_iter = referenced_special_tokens.keyIterator();
        while (referenced_iter.next()) |name| {
            if (!defined_special_tokens.contains(name.*)) return error.InvalidPostProcessor;
        }
    }
    return postprocessor;
}

fn mergeParsedPostProcessor(target: *schema.PostProcessor, sub: schema.PostProcessor) void {
    if (sub.type.len == 0 or std.mem.eql(u8, sub.type, "ByteLevel")) return;

    target.type = sub.type;
    target.add_special = target.add_special or sub.add_special;
    target.pair = target.pair or sub.pair;
    if (sub.cls_token != null) target.cls_token = sub.cls_token;
    if (sub.sep_token != null) target.sep_token = sub.sep_token;
}

/// Parse decoder section from tokenizer.json
/// Handles both Sequence decoder (with Strip) and simple decoders
fn mergeParsedDecoder(decoder: *schema.Decoder, sub: schema.Decoder) void {
    if (sub.strip_start != 0 or sub.strip_stop != 0) {
        decoder.strip_start = sub.strip_start;
        decoder.strip_stop = sub.strip_stop;
    }
    if (sub.metaspace) {
        decoder.metaspace = true;
        decoder.add_prefix_space = sub.add_prefix_space;
    }
    if (std.mem.eql(u8, sub.type, "WordPiece") or !sub.cleanup) {
        decoder.cleanup = sub.cleanup;
    }
}

fn parseDecoder(arena_allocator: std.mem.Allocator, json_scanner: *std.json.Scanner) !schema.Decoder {
    return parseDecoderDepth(arena_allocator, json_scanner, 0);
}

fn parseDecoderDepth(arena_allocator: std.mem.Allocator, json_scanner: *std.json.Scanner, depth: usize) !schema.Decoder {
    if (depth > MAX_JSON_PIPELINE_DEPTH) return error.InvalidDecoder;
    var decoder = schema.Decoder{};
    const first = try json_scanner.next();
    if (first == .null) return decoder;
    if (first != .object_begin) return error.InvalidDecoder;
    var saw_type = false;
    var saw_decoders = false;

    while (true) {
        const json_token = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
        switch (json_token) {
            .object_end => break,
            .allocated_string => |key| {
                if (std.mem.eql(u8, key, "type")) {
                    const type_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    decoder.type = switch (type_value) {
                        .allocated_string => |s| s,
                        .string => |s| s,
                        else => return error.InvalidDecoder,
                    };
                    saw_type = true;
                } else if (std.mem.eql(u8, key, "decoders")) {
                    // Sequence decoder - parse array of sub-decoders
                    if ((try json_scanner.next()) != .array_begin) return error.InvalidDecoder;
                    saw_decoders = true;
                    try parseDecoderSequence(arena_allocator, json_scanner, &decoder, depth + 1);
                } else if (std.mem.eql(u8, key, "start")) {
                    // Direct Strip decoder
                    const start_token = try json_scanner.next();
                    if (start_token == .number) decoder.strip_start = std.fmt.parseInt(i32, start_token.number, 10) catch 0;
                } else if (std.mem.eql(u8, key, "stop")) {
                    const stop_token = try json_scanner.next();
                    if (stop_token == .number) decoder.strip_stop = std.fmt.parseInt(i32, stop_token.number, 10) catch 0;
                } else if (std.mem.eql(u8, key, "add_prefix_space")) {
                    const val = try json_scanner.next();
                    if (val == .true) decoder.add_prefix_space = true;
                } else if (std.mem.eql(u8, key, "cleanup")) {
                    const val = try json_scanner.next();
                    decoder.cleanup = (val != .false);
                } else if (std.mem.eql(u8, key, "prepend_scheme")) {
                    const scheme_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    const scheme = switch (scheme_value) {
                        .allocated_string => |s| s,
                        .string => |s| s,
                        else => "",
                    };
                    if (std.mem.eql(u8, scheme, "first") or std.mem.eql(u8, scheme, "always")) {
                        decoder.add_prefix_space = true;
                    } else if (std.mem.eql(u8, scheme, "never")) {
                        decoder.add_prefix_space = false;
                    }
                } else {
                    try json_scanner.skipValue();
                }
            },
            else => return error.InvalidDecoder,
        }
    }
    if (!saw_type) return error.InvalidDecoder;
    if (std.mem.eql(u8, decoder.type, "Sequence") and !saw_decoders) return error.InvalidDecoder;
    if (std.mem.eql(u8, decoder.type, "Metaspace")) {
        decoder.metaspace = true;
    }
    if (!std.mem.eql(u8, decoder.type, "Sequence") and
        !std.mem.eql(u8, decoder.type, "Strip") and
        !std.mem.eql(u8, decoder.type, "ByteLevel") and
        !std.mem.eql(u8, decoder.type, "ByteFallback") and
        !std.mem.eql(u8, decoder.type, "Metaspace") and
        !std.mem.eql(u8, decoder.type, "WordPiece") and
        !std.mem.eql(u8, decoder.type, "BPEDecoder"))
    {
        return error.InvalidDecoder;
    }
    return decoder;
}

/// Parse decoders array within a Sequence decoder
fn parseDecoderSequence(arena_allocator: std.mem.Allocator, json_scanner: *std.json.Scanner, decoder: *schema.Decoder, depth: usize) !void {
    if (depth > MAX_JSON_PIPELINE_DEPTH) return error.InvalidDecoder;
    while (true) {
        const json_token = try json_scanner.next();
        switch (json_token) {
            .array_end => break,
            .object_begin => {
                // Parse sub-decoder object
                var sub = schema.Decoder{};
                var saw_type = false;
                var saw_decoders = false;
                var sub_pattern_string: ?[]const u8 = null;
                var sub_content: ?[]const u8 = null;
                while (true) {
                    const sub_token = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    switch (sub_token) {
                        .object_end => break,
                        .allocated_string => |key| {
                            if (std.mem.eql(u8, key, "type")) {
                                const type_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                                sub.type = switch (type_value) {
                                    .allocated_string => |s| s,
                                    .string => |s| s,
                                    else => return error.InvalidDecoder,
                                };
                                saw_type = true;
                            } else if (std.mem.eql(u8, key, "decoders")) {
                                if ((try json_scanner.next()) != .array_begin) return error.InvalidDecoder;
                                saw_decoders = true;
                                try parseDecoderSequence(arena_allocator, json_scanner, &sub, depth + 1);
                            } else if (std.mem.eql(u8, key, "start")) {
                                const start_token = try json_scanner.next();
                                if (start_token == .number) sub.strip_start = std.fmt.parseInt(i32, start_token.number, 10) catch 0;
                            } else if (std.mem.eql(u8, key, "stop")) {
                                const stop_token = try json_scanner.next();
                                if (stop_token == .number) sub.strip_stop = std.fmt.parseInt(i32, stop_token.number, 10) catch 0;
                            } else if (std.mem.eql(u8, key, "cleanup")) {
                                const cleanup_token = try json_scanner.next();
                                sub.cleanup = (cleanup_token != .false);
                            } else if (std.mem.eql(u8, key, "add_prefix_space")) {
                                const add_prefix_space_token = try json_scanner.next();
                                sub.add_prefix_space = (add_prefix_space_token == .true);
                            } else if (std.mem.eql(u8, key, "prepend_scheme")) {
                                const scheme_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                                const scheme = switch (scheme_value) {
                                    .allocated_string => |s| s,
                                    .string => |s| s,
                                    else => "",
                                };
                                if (std.mem.eql(u8, scheme, "first") or std.mem.eql(u8, scheme, "always")) {
                                    sub.add_prefix_space = true;
                                } else if (std.mem.eql(u8, scheme, "never")) {
                                    sub.add_prefix_space = false;
                                }
                            } else if (std.mem.eql(u8, key, "content")) {
                                const content_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                                sub_content = switch (content_value) {
                                    .allocated_string => |s| s,
                                    .string => |s| s,
                                    else => null,
                                };
                            } else if (std.mem.eql(u8, key, "pattern")) {
                                if ((try json_scanner.next()) != .object_begin) return error.InvalidDecoder;
                                while (true) {
                                    const pattern_token = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                                    switch (pattern_token) {
                                        .object_end => break,
                                        .allocated_string => |pattern_key| {
                                            if (std.mem.eql(u8, pattern_key, "String")) {
                                                const pattern_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                                                sub_pattern_string = switch (pattern_value) {
                                                    .allocated_string => |s| s,
                                                    .string => |s| s,
                                                    else => return error.InvalidDecoder,
                                                };
                                            } else {
                                                try json_scanner.skipValue();
                                            }
                                        },
                                        else => return error.InvalidDecoder,
                                    }
                                }
                            } else {
                                try json_scanner.skipValue();
                            }
                        },
                        else => return error.InvalidDecoder,
                    }
                }
                if (!saw_type) return error.InvalidDecoder;
                if (std.mem.eql(u8, sub.type, "Sequence") and !saw_decoders) return error.InvalidDecoder;
                if (std.mem.eql(u8, sub.type, "Metaspace")) {
                    sub.metaspace = true;
                } else if (std.mem.eql(u8, sub.type, "Replace")) {
                    if (sub_pattern_string) |pattern| {
                        if (sub_content) |content| {
                            if (std.mem.eql(u8, pattern, "▁") and std.mem.eql(u8, content, " ")) {
                                sub.metaspace = true;
                            }
                        }
                    }
                }
                if (!std.mem.eql(u8, sub.type, "Sequence") and
                    !std.mem.eql(u8, sub.type, "Strip") and
                    !std.mem.eql(u8, sub.type, "ByteLevel") and
                    !std.mem.eql(u8, sub.type, "Metaspace") and
                    !std.mem.eql(u8, sub.type, "WordPiece") and
                    !std.mem.eql(u8, sub.type, "BPEDecoder") and
                    !std.mem.eql(u8, sub.type, "Replace") and
                    !std.mem.eql(u8, sub.type, "Fuse") and
                    !std.mem.eql(u8, sub.type, "ByteFallback"))
                {
                    return error.InvalidDecoder;
                }
                mergeParsedDecoder(decoder, sub);
            },
            else => return error.InvalidDecoder,
        }
    }
}

// -------------------- Exports retained for C API --------------------

pub fn tokenizer_loader_from_json_string(json_data: ?[*:0]const u8) ?*ct.Tokenizer {
    const json_ptr = json_data orelse return null;
    const json_bytes = std.mem.sliceTo(json_ptr, 0);
    const allocator = std.heap.c_allocator;

    // Validate and parse once through the strict streaming loader first.
    // BPE still uses the lazy runtime model after validation to preserve
    // its encode-time behavior; WordPiece/Unigram build directly from root.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const root = load_from_slice_streaming(arena.allocator(), json_bytes) catch |err| {
        log.warn("tokenizer", "Tokenizer JSON parse failed", .{
            .reason = @errorName(err),
            .json_bytes = json_bytes.len,
        });
        return null;
    };
    const model_type_name = root.model.type;

    if (std.mem.eql(u8, model_type_name, "BPE")) {
        // Use lazy BPE loader (same path as file-based loading)
        // Need to copy the JSON since lazy loader may keep references to it
        const json_copy = allocator.dupeZ(u8, json_bytes) catch |err| {
            log.warn("tokenizer", "Tokenizer JSON copy failed", .{
                .reason = @errorName(err),
                .json_bytes = json_bytes.len,
                .model_type = model_type_name,
            });
            return null;
        };

        const tokenizer = ct.Tokenizer.initBpe(allocator, json_copy, true) catch {
            allocator.free(json_copy);
            log.warn("tokenizer", "Lazy BPE tokenizer init failed", .{
                .json_bytes = json_bytes.len,
            });
            return null;
        };

        // Keep the validated root-based metadata path for everything except
        // post-processing. TemplateProcessing must still use the dedicated
        // JSON parser to preserve one-sided BOS/EOS templates and Sequence
        // wrappers on the lazy BPE path.
        apply_root_metadata_without_postprocessor(&arena, tokenizer, root) catch |err| {
            tokenizer.destroy();
            log.warn("tokenizer", "Tokenizer metadata application failed", .{
                .reason = @errorName(err),
                .model_type = model_type_name,
            });
            return null;
        };
        const postprocessor_is_noop_sequence =
            std.mem.eql(u8, root.post_processor.type, "Sequence") and
            !root.post_processor.add_special and
            !root.post_processor.pair and
            root.post_processor.cls_token == null and
            root.post_processor.sep_token == null;

        if (postprocessor_is_noop_sequence) {
            apply_root_postprocessor_metadata(&arena, tokenizer, root) catch |err| {
                tokenizer.destroy();
                log.warn("tokenizer", "Tokenizer post-processor application failed", .{
                    .reason = @errorName(err),
                    .model_type = model_type_name,
                });
                return null;
            };
        } else if (findSection(json_bytes, "\"post_processor\"")) |section| {
            if (section.len > 0 and section[0] == '{') {
                const end = findMatchingBrace(section, '{', '}') orelse section.len;
                applyPostProcessorFromJson(tokenizer, section[0..end]) catch |err| {
                    tokenizer.destroy();
                    log.warn("tokenizer", "Tokenizer post-processor application failed", .{
                        .reason = @errorName(err),
                        .model_type = model_type_name,
                    });
                    return null;
                };
            }
        }
        apply_root_decoder_metadata(tokenizer, root);
        return @ptrCast(tokenizer);
    }

    return build_tokenizer_from_root(&arena, root) catch |err| {
        log.warn("tokenizer", "Tokenizer model build failed", .{
            .reason = @errorName(err),
            .model_type = model_type_name,
        });
        return null;
    };
}

/// Find the snapshot directory for cache model layout (models--org--name/snapshots/)
fn findSnapshotDir(ally: std.mem.Allocator, base_path: []const u8) ?[]const u8 {
    const snapshots_path = std.fs.path.join(ally, &.{ base_path, "snapshots" }) catch return null;
    defer ally.free(snapshots_path);

    // First try refs/main to get the canonical revision
    const refs_main_path = std.fs.path.join(ally, &.{ base_path, "refs", "main" }) catch return null;
    defer ally.free(refs_main_path);

    if (std.fs.cwd().openFile(refs_main_path, .{})) |file| {
        defer file.close();
        var rev_buf: [256]u8 = undefined;
        const read_len = file.read(&rev_buf) catch 0;
        if (read_len > 0) {
            // Trim whitespace/newlines
            var end = read_len;
            while (end > 0 and (rev_buf[end - 1] == '\n' or rev_buf[end - 1] == '\r' or rev_buf[end - 1] == ' ')) {
                end -= 1;
            }
            if (end > 0) {
                const rev = rev_buf[0..end];
                const candidate = std.fs.path.join(ally, &.{ snapshots_path, rev }) catch return null;
                // Check if directory exists using access
                if (std.fs.cwd().access(candidate, .{})) |_| {
                    return candidate;
                } else |_| {
                    ally.free(candidate);
                }
            }
        }
    } else |_| {}

    // Not in cache - iterate snapshots directory (not available on WASM/Emscripten)
    const builtin = @import("builtin");
    if (comptime builtin.target.os.tag == .emscripten or builtin.target.os.tag == .wasi) {
        return null;
    }

    var snapshots_dir = std.fs.cwd().openDir(snapshots_path, .{ .iterate = true }) catch return null;
    defer snapshots_dir.close();

    var iter = snapshots_dir.iterate();
    while (iter.next() catch null) |entry| {
        if (entry.kind == .directory) {
            const candidate = std.fs.path.join(ally, &.{ snapshots_path, entry.name }) catch continue;
            return candidate;
        }
    }

    return null;
}

pub fn tokenizer_loader_from_dir(path: ?[*:0]const u8) ?*ct.Tokenizer {
    const path_ptr = path orelse return null;
    const dir_bytes = std.mem.sliceTo(path_ptr, 0);
    const allocator = std.heap.c_allocator;
    var t_start: i128 = std.time.nanoTimestamp();

    var path_buf: [512]u8 = undefined; // filled by bufPrint before use
    const json_path = blk: {
        // If path already ends with tokenizer.json, use it directly
        if (std.mem.endsWith(u8, dir_bytes, "tokenizer.json")) {
            if (std.fs.cwd().access(dir_bytes, .{})) |_| {
                break :blk dir_bytes;
            } else |_| {}
        }

        // Try direct path first
        const direct_len = std.fmt.bufPrint(&path_buf, "{s}/tokenizer.json", .{dir_bytes}) catch return null;
        if (std.fs.cwd().access(path_buf[0..direct_len.len], .{})) |_| {
            break :blk path_buf[0..direct_len.len];
        } else |_| {}

        // Try cache snapshot layout
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        if (findSnapshotDir(arena.allocator(), dir_bytes)) |snapshot_dir| {
            const snap_len = std.fmt.bufPrint(&path_buf, "{s}/tokenizer.json", .{snapshot_dir}) catch return null;
            break :blk path_buf[0..snap_len.len];
        }
        return null;
    };

    // Read JSON file
    var file = std.fs.cwd().openFile(json_path, .{}) catch return null;
    defer file.close();
    const stat = file.stat() catch return null;
    const json_len: usize = @intCast(stat.size);
    const json_bytes = allocator.alloc(u8, json_len) catch return null;
    errdefer allocator.free(json_bytes);
    const bytes_read = file.readAll(json_bytes) catch {
        allocator.free(json_bytes);
        return null;
    };
    if (bytes_read != json_len) {
        allocator.free(json_bytes);
        return null;
    }

    {
        const now = std.time.nanoTimestamp();
        const duration_ms = @as(f64, @floatFromInt(now - t_start)) / 1_000_000.0;
        log.debug("tokenizer", "Read JSON", .{ .size_kb = json_len / 1024, .duration_ms = duration_ms }, @src());
        t_start = now;
    }

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const root = load_from_slice_streaming(arena.allocator(), json_bytes) catch |err| {
        log.warn("tokenizer", "Tokenizer JSON parse failed", .{
            .reason = @errorName(err),
            .json_path = json_path,
            .json_bytes = json_bytes.len,
        });
        allocator.free(json_bytes);
        return null;
    };
    const model_type_name = root.model.type;

    if (std.mem.eql(u8, model_type_name, "BPE")) {
        // Use lazy BPE loader - defers vocab/merges parsing until first encode
        const tokenizer = ct.Tokenizer.initBpe(allocator, json_bytes, true) catch {
            log.warn("tokenizer", "Lazy BPE tokenizer init failed", .{
                .json_path = json_path,
                .json_bytes = json_bytes.len,
            });
            allocator.free(json_bytes);
            return null;
        };
        {
            const now = std.time.nanoTimestamp();
            const duration_ms = @as(f64, @floatFromInt(now - t_start)) / 1_000_000.0;
            log.debug("tokenizer", "Lazy init", .{ .duration_ms = duration_ms }, @src());
            t_start = now;
        }

        // Keep the validated root-based metadata path for everything except
        // post-processing. TemplateProcessing must still use the dedicated
        // JSON parser to preserve one-sided BOS/EOS templates and Sequence
        // wrappers on the lazy BPE path.
        apply_root_metadata_without_postprocessor(&arena, tokenizer, root) catch |err| {
            tokenizer.destroy();
            log.warn("tokenizer", "Tokenizer metadata application failed", .{
                .reason = @errorName(err),
                .json_path = json_path,
                .model_type = model_type_name,
            });
            return null;
        };
        if (findSection(json_bytes, "\"post_processor\"")) |section| {
            if (section.len > 0 and section[0] == '{') {
                const end = findMatchingBrace(section, '{', '}') orelse section.len;
                applyPostProcessorFromJson(tokenizer, section[0..end]) catch |err| {
                    tokenizer.destroy();
                    log.warn("tokenizer", "Tokenizer post-processor application failed", .{
                        .reason = @errorName(err),
                        .json_path = json_path,
                        .model_type = model_type_name,
                    });
                    return null;
                };
            }
        }
        apply_root_decoder_metadata(tokenizer, root);
        {
            const now = std.time.nanoTimestamp();
            const duration_ms = @as(f64, @floatFromInt(now - t_start)) / 1_000_000.0;
            log.debug("tokenizer", "Apply config", .{ .duration_ms = duration_ms }, @src());
        }
        return @ptrCast(tokenizer);
    }

    // json_bytes must stay alive until after build_tokenizer_from_root — the
    // parsed root struct contains slices that reference the original buffer
    // (e.g. model.type from detectModelType, normalizer/pretokenizer fields
    // from std.json Scanner's .alloc_if_needed path).
    defer allocator.free(json_bytes);
    {
        const now = std.time.nanoTimestamp();
        const duration_ms = @as(f64, @floatFromInt(now - t_start)) / 1_000_000.0;
        log.debug("tokenizer", "Parse JSON", .{ .duration_ms = duration_ms }, @src());
        t_start = now;
    }
    const result = build_tokenizer_from_root(&arena, root) catch |err| {
        log.warn("tokenizer", "Tokenizer model build failed", .{
            .reason = @errorName(err),
            .json_path = json_path,
            .model_type = model_type_name,
        });
        return null;
    };
    {
        const now = std.time.nanoTimestamp();
        const duration_ms = @as(f64, @floatFromInt(now - t_start)) / 1_000_000.0;
        log.debug("tokenizer", "Build model", .{ .duration_ms = duration_ms }, @src());
    }
    return result;
}

/// Detect model type by scanning JSON for "type" field in "model" section
fn detectModelType(json_bytes: []const u8) []const u8 {
    // Find "model" section
    if (std.mem.indexOf(u8, json_bytes, "\"model\"")) |model_pos| {
        // Find "type" within next 200 bytes
        const search_end = @min(model_pos + 200, json_bytes.len);
        if (std.mem.indexOf(u8, json_bytes[model_pos..search_end], "\"type\"")) |type_pos| {
            const abs_pos = model_pos + type_pos + 6; // skip "type"
            // Find the value string
            var cursor = abs_pos;
            while (cursor < json_bytes.len and json_bytes[cursor] != '"') : (cursor += 1) {}
            if (cursor >= json_bytes.len) return "BPE";
            cursor += 1;
            const value_start = cursor;
            while (cursor < json_bytes.len and json_bytes[cursor] != '"') : (cursor += 1) {}
            return json_bytes[value_start..cursor];
        }
    }
    return "BPE";
}

/// Apply config from JSON (added_tokens, normalizer, pre_tokenizer, post_processor)
/// This is fast - these sections are small
fn applyConfigFromJson(tokenizer_any: anytype, json_bytes: []const u8) !void {
    const tokenizer: *ct.Tokenizer = @ptrCast(tokenizer_any);
    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    // Parse added_tokens
    if (findSection(json_bytes, "\"added_tokens\"")) |section| {
        if (section.len > 0 and section[0] == '[') {
            const end = findMatchingBrace(section, '[', ']') orelse section.len;
            try parseAndApplyAddedTokens(tokenizer, section[0..end], arena_allocator);
        }
    }

    // Parse normalizer
    if (findSection(json_bytes, "\"normalizer\"")) |section| {
        if (section.len > 0 and section[0] == '{') {
            const end = findMatchingBrace(section, '{', '}') orelse section.len;
            var scanner = std.json.Scanner.initCompleteInput(arena_allocator, section[0..end]);
            const normalizer = try parseNormalizer(arena_allocator, &scanner);
            const normalizer_spec = ct.NormalizerSpec{
                .type = if (normalizer.type.len > 0) (std.heap.c_allocator.dupeZ(u8, normalizer.type) catch return error.OutOfMemory).ptr else null,
                .lowercase = if (normalizer.lowercase) 1 else 0,
                .strip_accents = if (normalizer.strip_accents) 1 else 0,
                .nfc = if (normalizer.nfc) 1 else 0,
                .nfd = if (normalizer.nfd) 1 else 0,
                .nfkc = if (normalizer.nfkc) 1 else 0,
                .nfkd = if (normalizer.nfkd) 1 else 0,
                .clean_text = if (normalizer.clean_text) 1 else 0,
                .handle_chinese_chars = if (normalizer.handle_chinese_chars) 1 else 0,
                .prepend = if (normalizer.prepend) |p| (std.heap.c_allocator.dupeZ(u8, p) catch return error.OutOfMemory).ptr else null,
                .replace_pattern = if (normalizer.replace_pattern) |p| (std.heap.c_allocator.dupeZ(u8, p) catch return error.OutOfMemory).ptr else null,
                .replace_content = if (normalizer.replace_content) |c_val| (std.heap.c_allocator.dupeZ(u8, c_val) catch return error.OutOfMemory).ptr else null,
            };
            tok_fns.tokenizer_apply_normalizer_spec(tokenizer, &normalizer_spec);
        }
    }

    // Parse pre_tokenizer
    if (findSection(json_bytes, "\"pre_tokenizer\"")) |section| {
        if (section.len > 0 and section[0] == '{') {
            const end = findMatchingBrace(section, '{', '}') orelse section.len;
            var scanner = std.json.Scanner.initCompleteInput(arena_allocator, section[0..end]);
            const pretokenizer = try parsePreTokenizer(arena_allocator, &scanner);
            const pretokenizer_spec = ct.PreTokenizerSpec{
                .type = if (pretokenizer.type.len > 0) (std.heap.c_allocator.dupeZ(u8, pretokenizer.type) catch return error.OutOfMemory).ptr else null,
                .add_prefix_space = if (pretokenizer.add_prefix_space) 1 else 0,
                .trim_offsets = if (pretokenizer.trim_offsets) 1 else 0,
                .use_regex = if (pretokenizer.use_regex) 1 else 0,
                .byte_level = if (pretokenizer.byte_level) 1 else 0,
                .whitespace = if (pretokenizer.whitespace) 1 else 0,
                .punctuation = if (pretokenizer.punctuation) 1 else 0,
                .pattern = if (pretokenizer.pattern) |p| (std.heap.c_allocator.dupeZ(u8, p) catch return error.OutOfMemory).ptr else null,
                .regex_split = if (pretokenizer.regex_split) 1 else 0,
                .regex_invert = if (pretokenizer.regex_invert) 1 else 0,
                .metaspace = if (pretokenizer.metaspace) 1 else 0,
            };
            tok_fns.tokenizer_apply_pretokenizer_spec(tokenizer, &pretokenizer_spec);
        }
    }

    // Parse post_processor
    if (findSection(json_bytes, "\"post_processor\"")) |section| {
        if (section.len > 0 and section[0] == '{') {
            const end = findMatchingBrace(section, '{', '}') orelse section.len;
            try applyPostProcessorFromJson(tokenizer, section[0..end]);
        }
    }

    // Parse decoder
    if (findSection(json_bytes, "\"decoder\"")) |section| {
        if (section.len > 0 and section[0] == '{') {
            const end = findMatchingBrace(section, '{', '}') orelse section.len;
            applyDecoderFromJson(tokenizer, section[0..end], arena_allocator);
        }
    }
}

/// Parse added_tokens and apply directly
fn parseAndApplyAddedTokens(tokenizer: *ct.Tokenizer, json_bytes: []const u8, _: std.mem.Allocator) !void {
    var cursor: usize = 0;
    while (cursor < json_bytes.len) {
        // Find start of token object
        while (cursor < json_bytes.len and json_bytes[cursor] != '{') : (cursor += 1) {}
        if (cursor >= json_bytes.len) break;

        const obj_start = cursor;
        const obj_end = if (findMatchingBrace(json_bytes[cursor..], '{', '}')) |len| cursor + len else break;

        const obj = json_bytes[obj_start..obj_end];

        // Extract id
        var token_id: i32 = 0;
        if (findJsonFieldValue(obj, "\"id\"")) |id_str| {
            token_id = std.fmt.parseInt(i32, id_str, 10) catch 0;
        }

        // Extract content
        var token_content: []const u8 = "";
        if (findJsonFieldString(obj, "\"content\"")) |content_str| {
            token_content = content_str;
        }

        // Extract special flag
        var special_flag: c_int = 0;
        if (findJsonFieldValue(obj, "\"special\"")) |special_str| {
            special_flag = if (std.mem.eql(u8, special_str, "true")) 1 else 0;
        }

        if (std.mem.indexOfScalar(u8, token_content, 0) != null) return error.InvalidAdded;

        // Add token
        // Use c_allocator for persistent allocations (arena is freed on function return)
        if (token_content.len > 0) {
            // Unescape JSON escape sequences (\t, \n, \\, etc.) and null-terminate.
            // Unescaping can only shorten the string, so token_content.len is the max size.
            const content_dup = std.heap.c_allocator.allocSentinel(u8, token_content.len, 0) catch return error.OutOfMemory;
            var out_i: usize = 0;
            var in_i: usize = 0;
            while (in_i < token_content.len) {
                if (token_content[in_i] == '\\' and in_i + 1 < token_content.len) {
                    content_dup[out_i] = switch (token_content[in_i + 1]) {
                        'n' => '\n',
                        'r' => '\r',
                        't' => '\t',
                        'b' => 0x08,
                        'f' => 0x0C,
                        '\\' => '\\',
                        '"' => '"',
                        '/' => '/',
                        else => token_content[in_i + 1],
                    };
                    in_i += 2;
                } else {
                    content_dup[out_i] = token_content[in_i];
                    in_i += 1;
                }
                out_i += 1;
            }
            content_dup[out_i] = 0;
            const added_node = tok_fns.tokenizer_added_token_add(tokenizer, content_dup.ptr, token_id, special_flag);
            if (added_node != null) {
                // Parse additional flags
                if (findJsonFieldValue(obj, "\"single_word\"")) |v| {
                    added_node.?.single_word = if (std.mem.eql(u8, v, "true")) 1 else 0;
                }
                if (findJsonFieldValue(obj, "\"lstrip\"")) |v| {
                    added_node.?.lstrip = if (std.mem.eql(u8, v, "true")) 1 else 0;
                }
                if (findJsonFieldValue(obj, "\"rstrip\"")) |v| {
                    added_node.?.rstrip = if (std.mem.eql(u8, v, "true")) 1 else 0;
                }
                if (findJsonFieldValue(obj, "\"normalized\"")) |v| {
                    added_node.?.normalized = if (std.mem.eql(u8, v, "true")) 1 else 0;
                }
            }
        }

        cursor = obj_end;
    }
}

/// Find a field value (number/bool) in JSON object
fn findJsonFieldValue(json_bytes: []const u8, field_bytes: []const u8) ?[]const u8 {
    const pos = findJsonKeyPos(json_bytes, field_bytes) orelse return null;
    var cursor = pos + field_bytes.len;
    while (cursor < json_bytes.len and (json_bytes[cursor] == ' ' or json_bytes[cursor] == ':' or json_bytes[cursor] == '\t' or json_bytes[cursor] == '\n' or json_bytes[cursor] == '\r')) : (cursor += 1) {}
    if (cursor >= json_bytes.len) return null;
    const value_start = cursor;
    while (cursor < json_bytes.len and json_bytes[cursor] != ',' and json_bytes[cursor] != '}' and json_bytes[cursor] != ']' and json_bytes[cursor] != ' ' and json_bytes[cursor] != '\n' and json_bytes[cursor] != '\r') : (cursor += 1) {}
    return json_bytes[value_start..cursor];
}

fn findJsonFieldString(json_bytes: []const u8, field_bytes: []const u8) ?[]const u8 {
    const pos = findJsonKeyPos(json_bytes, field_bytes) orelse return null;
    var cursor = pos + field_bytes.len;
    while (cursor < json_bytes.len and (json_bytes[cursor] == ' ' or json_bytes[cursor] == ':' or json_bytes[cursor] == '\t' or json_bytes[cursor] == '\n' or json_bytes[cursor] == '\r')) : (cursor += 1) {}
    if (cursor >= json_bytes.len or json_bytes[cursor] != '"') return null;
    cursor += 1;
    const value_start = cursor;
    while (cursor < json_bytes.len and json_bytes[cursor] != '"') {
        if (json_bytes[cursor] == '\\') cursor += 2 else cursor += 1;
    }
    return json_bytes[value_start..cursor];
}

/// Parse a JSON field whose value is a ["string", number] tuple (e.g. BertProcessing cls/sep).
/// Returns the string element, or null if the field is absent or not in array format.
fn findJsonFieldArrayString(json_bytes: []const u8, field_bytes: []const u8) ?[]const u8 {
    const pos = findJsonKeyPos(json_bytes, field_bytes) orelse return null;
    var cursor = pos + field_bytes.len;
    // Skip whitespace and colon
    while (cursor < json_bytes.len and (json_bytes[cursor] == ' ' or json_bytes[cursor] == ':' or json_bytes[cursor] == '\t' or json_bytes[cursor] == '\n')) : (cursor += 1) {}
    if (cursor >= json_bytes.len or json_bytes[cursor] != '[') return null;
    cursor += 1;
    // Skip whitespace inside array
    while (cursor < json_bytes.len and (json_bytes[cursor] == ' ' or json_bytes[cursor] == '\t' or json_bytes[cursor] == '\n')) : (cursor += 1) {}
    if (cursor >= json_bytes.len or json_bytes[cursor] != '"') return null;
    cursor += 1;
    const str_start = cursor;
    while (cursor < json_bytes.len and json_bytes[cursor] != '"') {
        if (json_bytes[cursor] == '\\') cursor += 2 else cursor += 1;
    }
    return json_bytes[str_start..cursor];
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
            if (key_end - key_start == field_bytes.len and std.mem.eql(u8, json_bytes[key_start..key_end], field_bytes)) {
                return key_start;
            }
        } else {
            cursor += 1;
        }
    }
    return null;
}

fn applyNormalizerFromJson(tokenizer: *ct.Tokenizer, json_bytes: []const u8, arena_allocator: std.mem.Allocator) !void {
    try ensureJsonDepthWithinLimit(json_bytes, error.InvalidNormalizer);
    return applyNormalizerFromJsonDepth(tokenizer, json_bytes, arena_allocator, 0);
}

fn applyNormalizerFromJsonDepth(tokenizer: *ct.Tokenizer, json_bytes: []const u8, arena_allocator: std.mem.Allocator, depth: usize) !void {
    if (depth > MAX_JSON_PIPELINE_DEPTH) return error.InvalidNormalizer;
    // Handle "type" field to infer normalization type
    if (findJsonFieldValue(json_bytes, "\"type\"")) |type_val| {
        // Type values are quoted strings like "NFC", "NFD", etc.
        // Strip surrounding quotes if present
        const type_str = if (type_val.len >= 2 and type_val[0] == '"')
            type_val[1 .. type_val.len - 1]
        else
            type_val;
        if (std.mem.eql(u8, type_str, "NFC")) {
            tokenizer.*.normalizer.nfc = 1;
        } else if (std.mem.eql(u8, type_str, "NFD")) {
            tokenizer.*.normalizer.nfd = 1;
        } else if (std.mem.eql(u8, type_str, "NFKC")) {
            tokenizer.*.normalizer.nfkc = 1;
        } else if (std.mem.eql(u8, type_str, "NFKD")) {
            tokenizer.*.normalizer.nfkd = 1;
        } else if (std.mem.eql(u8, type_str, "Lowercase")) {
            tokenizer.*.normalizer.lowercase = 1;
        } else if (std.mem.eql(u8, type_str, "StripAccents")) {
            tokenizer.*.normalizer.strip_accents = 1;
        } else if (std.mem.eql(u8, type_str, "BertNormalizer")) {
            tokenizer.*.normalizer.lowercase = 1;
            tokenizer.*.normalizer.strip_accents = 1;
            tokenizer.*.normalizer.clean_text = 1;
            tokenizer.*.normalizer.handle_chinese_chars = 1;
        } else if (std.mem.eql(u8, type_str, "Sequence")) {
            // Handle Sequence normalizer - process nested normalizers
            if (findSection(json_bytes, "\"normalizers\"")) |arr_section| {
                if (arr_section.len > 0 and arr_section[0] == '[') {
                    const arr_end = findMatchingBrace(arr_section, '[', ']') orelse arr_section.len;
                    const arr_content = arr_section[0..arr_end];
                    // Recursively apply each normalizer in the sequence
                    var cursor: usize = 0;
                    while (cursor < arr_content.len) {
                        // Find start of next object
                        if (std.mem.indexOfPos(u8, arr_content, cursor, "{")) |obj_start| {
                            const obj_end = findMatchingBrace(arr_content[obj_start..], '{', '}') orelse break;
                            const obj_content = arr_content[obj_start .. obj_start + obj_end];
                            try applyNormalizerFromJsonDepth(tokenizer, obj_content, arena_allocator, depth + 1);
                            cursor = obj_start + obj_end;
                        } else break;
                    }
                }
            }
            return; // Don't process other fields for Sequence type
        } else if (std.mem.eql(u8, type_str, "Prepend")) {
            // Handle Prepend normalizer
            // Use c_allocator for persistent allocations (arena is freed on function return)
            if (findJsonFieldString(json_bytes, "\"prepend\"")) |prepend_str| {
                const unescaped = try json_utils.unescapeJsonString(arena_allocator, prepend_str);
                const dup = std.heap.c_allocator.dupeZ(u8, unescaped) catch return error.OutOfMemory;
                tokenizer.*.normalizer.prepend = dup.ptr;
            }
            return;
        } else if (std.mem.eql(u8, type_str, "Replace")) {
            // Handle Replace normalizer
            // Pattern can be {"String": "..."} - find the String value
            // Use c_allocator for persistent allocations (arena is freed on function return)
            if (findSection(json_bytes, "\"pattern\"")) |pat_section| {
                if (findJsonFieldString(pat_section, "\"String\"")) |pat_str| {
                    const unescaped = try json_utils.unescapeJsonString(arena_allocator, pat_str);
                    const dup = std.heap.c_allocator.dupeZ(u8, unescaped) catch return error.OutOfMemory;
                    tokenizer.*.normalizer.replace_pattern = dup.ptr;
                }
            }
            if (findJsonFieldString(json_bytes, "\"content\"")) |content_str| {
                const unescaped = try json_utils.unescapeJsonString(arena_allocator, content_str);
                const dup = std.heap.c_allocator.dupeZ(u8, unescaped) catch return error.OutOfMemory;
                tokenizer.*.normalizer.replace_content = dup.ptr;
            }
            return;
        }
    }
    if (findJsonFieldValue(json_bytes, "\"lowercase\"")) |v| {
        tokenizer.*.normalizer.lowercase = if (std.mem.eql(u8, v, "true")) 1 else 0;
    }
    if (findJsonFieldValue(json_bytes, "\"strip_accents\"")) |v| {
        tokenizer.*.normalizer.strip_accents = if (std.mem.eql(u8, v, "true")) 1 else 0;
    }
    if (findJsonFieldValue(json_bytes, "\"nfc\"")) |v| {
        tokenizer.*.normalizer.nfc = if (std.mem.eql(u8, v, "true")) 1 else 0;
    }
    if (findJsonFieldValue(json_bytes, "\"nfd\"")) |v| {
        tokenizer.*.normalizer.nfd = if (std.mem.eql(u8, v, "true")) 1 else 0;
    }
    if (findJsonFieldValue(json_bytes, "\"nfkc\"")) |v| {
        tokenizer.*.normalizer.nfkc = if (std.mem.eql(u8, v, "true")) 1 else 0;
    }
    if (findJsonFieldValue(json_bytes, "\"clean_text\"")) |v| {
        tokenizer.*.normalizer.clean_text = if (std.mem.eql(u8, v, "true")) 1 else 0;
    }
    if (findJsonFieldValue(json_bytes, "\"handle_chinese_chars\"")) |v| {
        tokenizer.*.normalizer.handle_chinese_chars = if (std.mem.eql(u8, v, "true")) 1 else 0;
    }
}

fn applyPreTokenizerFromJson(tokenizer: *ct.Tokenizer, json_bytes: []const u8, arena_allocator: std.mem.Allocator) !void {
    try ensureJsonDepthWithinLimit(json_bytes, error.InvalidPreTokenizer);
    return applyPreTokenizerFromJsonDepth(tokenizer, json_bytes, arena_allocator, 0);
}

fn applyPreTokenizerFromJsonDepth(tokenizer: *ct.Tokenizer, json_bytes: []const u8, arena_allocator: std.mem.Allocator, depth: usize) !void {
    if (depth > MAX_JSON_PIPELINE_DEPTH) return error.InvalidPreTokenizer;
    if (findJsonFieldValue(json_bytes, "\"add_prefix_space\"")) |v| {
        tokenizer.*.pretokenizer.add_prefix_space = if (std.mem.eql(u8, v, "true")) 1 else 0;
    }
    // Newer HF config uses prepend_scheme instead of add_prefix_space (Metaspace)
    if (findJsonFieldString(json_bytes, "\"prepend_scheme\"")) |scheme| {
        if (std.mem.eql(u8, scheme, "first") or std.mem.eql(u8, scheme, "always")) {
            tokenizer.*.pretokenizer.add_prefix_space = 1;
        } else if (std.mem.eql(u8, scheme, "never")) {
            tokenizer.*.pretokenizer.add_prefix_space = 0;
        }
    }
    if (findJsonFieldValue(json_bytes, "\"trim_offsets\"")) |v| {
        tokenizer.*.pretokenizer.trim_offsets = if (std.mem.eql(u8, v, "true")) 1 else 0;
    }
    // Check for type
    if (findJsonFieldString(json_bytes, "\"type\"")) |type_str| {
        if (std.mem.eql(u8, type_str, "Sequence")) {
            // Sequence type - process nested pretokenizers
            if (findSection(json_bytes, "\"pretokenizers\"")) |arr_section| {
                if (arr_section.len > 0 and arr_section[0] == '[') {
                    const arr_end = findMatchingBrace(arr_section, '[', ']') orelse arr_section.len;
                    const arr_content = arr_section[0..arr_end];
                    // Find each pretokenizer object in the array
                    var cursor: usize = 0;
                    while (cursor < arr_content.len) {
                        while (cursor < arr_content.len and arr_content[cursor] != '{') : (cursor += 1) {}
                        if (cursor >= arr_content.len) break;
                        const obj_start = cursor;
                        const obj_end = if (findMatchingBrace(arr_content[cursor..], '{', '}')) |len| cursor + len else break;
                        const obj = arr_content[obj_start..obj_end];
                        // Recursively apply this nested pretokenizer
                        try applyPreTokenizerFromJsonDepth(tokenizer, obj, arena_allocator, depth + 1);
                        cursor = obj_end;
                    }
                }
            }
        } else if (std.mem.eql(u8, type_str, "ByteLevel")) {
            tokenizer.*.pretokenizer.byte_level = 1;
            // ByteLevel with use_regex=true needs its own GPT-2 regex for splitting.
            // This overrides any prior regex (e.g., a no-op \Q...\E from a String Split).
            if (findJsonFieldValue(json_bytes, "\"use_regex\"")) |v| {
                if (std.mem.eql(u8, v, "true")) {
                    const gpt2_pattern: [*:0]const u8 = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
                    _ = tok_fns.tokenizer_pretokenizer_set(&tokenizer.*.pretokenizer, gpt2_pattern);
                } else if (std.mem.eql(u8, v, "false")) {
                    tok_fns.tokenizer_pretokenizer_free(&tokenizer.*.pretokenizer);
                    tokenizer.*.pretokenizer.regex_split = 0;
                    tokenizer.*.pretokenizer.regex_invert = 0;
                }
            }
        } else if (std.mem.eql(u8, type_str, "Whitespace") or std.mem.eql(u8, type_str, "WhitespaceSplit")) {
            _ = tok_fns.tokenizer_pretokenizer_set(&tokenizer.*.pretokenizer, null);
            tokenizer.*.pretokenizer.whitespace = 1;
        } else if (std.mem.eql(u8, type_str, "Punctuation")) {
            _ = tok_fns.tokenizer_pretokenizer_set(&tokenizer.*.pretokenizer, null);
            tokenizer.*.pretokenizer.punctuation = 1;
        } else if (std.mem.eql(u8, type_str, "BertPreTokenizer")) {
            // Clear model default regex — BertPreTokenizer uses whitespace/punctuation splitting
            _ = tok_fns.tokenizer_pretokenizer_set(&tokenizer.*.pretokenizer, null);
            tokenizer.*.pretokenizer.whitespace = 1;
            tokenizer.*.pretokenizer.punctuation = 1;
        } else if (std.mem.eql(u8, type_str, "Metaspace")) {
            // Clear BPE default regex — Metaspace uses whitespace splitting, not regex
            _ = tok_fns.tokenizer_pretokenizer_set(&tokenizer.*.pretokenizer, null);
            tokenizer.*.pretokenizer.metaspace = 1;
            tokenizer.*.pretokenizer.whitespace = 1;
        } else if (std.mem.eql(u8, type_str, "Split")) {
            // Split type - parse pattern, behavior, and invert
            // Behavior: "Isolated" = emit matches and gaps, "Removed" = drop matches
            //           "MergedWithPrevious/Next" = split on pattern
            // Note: Don't reset byte_level here - it may be set by a sibling ByteLevel pretokenizer in a Sequence

            // Check behavior - default to MergedWithPrevious (regex_split=1)
            // "Isolated" emits matches and gaps; everything else splits on pattern.
            if (findJsonFieldString(json_bytes, "\"behavior\"")) |behavior| {
                if (std.mem.eql(u8, behavior, "Isolated")) {
                    tokenizer.*.pretokenizer.regex_split = 0;
                } else {
                    tokenizer.*.pretokenizer.regex_split = 1;
                }
            } else {
                tokenizer.*.pretokenizer.regex_split = 1;
            }

            // Check invert - if true, emit matches instead of gaps
            // With invert=true, the regex pattern describes what tokens to KEEP
            if (findJsonFieldValue(json_bytes, "\"invert\"")) |invert_val| {
                if (std.mem.eql(u8, invert_val, "true")) {
                    tokenizer.*.pretokenizer.regex_invert = 1;
                    tokenizer.*.pretokenizer.regex_split = 0;
                }
            }

            // Parse pattern - can be {"String": " "} or {"Regex": "..."}
            // Use c_allocator for persistent allocations (arena is freed on function return)
            if (findSection(json_bytes, "\"pattern\"")) |pattern_section| {
                if (pattern_section.len > 0 and pattern_section[0] == '{') {
                    // Object format: {"String": " "} or {"Regex": "..."}
                    if (findJsonFieldString(pattern_section, "\"String\"")) |str_pattern| {
                        // String patterns are literal matches — wrap with \Q...\E so
                        // PCRE2 treats the value as a literal, not as a regex.
                        const unescaped = try json_utils.unescapeJsonString(arena_allocator, str_pattern);
                        const content_len = 2 + unescaped.len + 2; // \Q + content + \E
                        const pat_z = std.heap.c_allocator.allocSentinel(u8, content_len, 0) catch return error.OutOfMemory;
                        pat_z[0] = '\\';
                        pat_z[1] = 'Q';
                        @memcpy(pat_z[2 .. 2 + unescaped.len], unescaped);
                        pat_z[2 + unescaped.len] = '\\';
                        pat_z[2 + unescaped.len + 1] = 'E';
                        _ = tok_fns.tokenizer_pretokenizer_set(&tokenizer.*.pretokenizer, pat_z.ptr);
                    } else if (findJsonFieldString(pattern_section, "\"Regex\"")) |regex_pattern| {
                        // Unescape JSON string and compile pattern as regex
                        const unescaped = try json_utils.unescapeJsonString(arena_allocator, regex_pattern);
                        const pat_z = std.heap.c_allocator.dupeZ(u8, unescaped) catch return error.OutOfMemory;
                        _ = tok_fns.tokenizer_pretokenizer_set(&tokenizer.*.pretokenizer, pat_z.ptr);
                    }
                }
            }
        }
    }
}

/// Apply decoder settings from JSON (fast path)
/// Apply post_processor settings from JSON for the fast-path BPE loader.
/// Parses type, cls/sep tokens (BertProcessing format), and resolves IDs
/// from the added_tokens list.
fn applyPostProcessorFromJson(tokenizer: *ct.Tokenizer, json_bytes: []const u8) !void {
    try ensureJsonDepthWithinLimit(json_bytes, error.InvalidPostProcessor);
    return applyPostProcessorFromJsonDepth(tokenizer, json_bytes, 0);
}

fn applyPostProcessorFromJsonDepth(tokenizer: *ct.Tokenizer, json_bytes: []const u8, depth: usize) !void {
    if (depth > MAX_JSON_PIPELINE_DEPTH) return error.InvalidPostProcessor;
    const type_str = findJsonFieldString(json_bytes, "\"type\"") orelse return;

    // Sequence type: iterate processors array and apply each sub-processor
    if (std.mem.eql(u8, type_str, "Sequence")) {
        if (findSection(json_bytes, "\"processors\"")) |arr_section| {
            if (arr_section.len > 0 and arr_section[0] == '[') {
                const arr_end = findMatchingBrace(arr_section, '[', ']') orelse arr_section.len;
                const arr_content = arr_section[0..arr_end];
                var cursor: usize = 0;
                while (cursor < arr_content.len) {
                    while (cursor < arr_content.len and arr_content[cursor] != '{') : (cursor += 1) {}
                    if (cursor >= arr_content.len) break;
                    const obj_end = if (findMatchingBrace(arr_content[cursor..], '{', '}')) |len| cursor + len else break;
                    try applyPostProcessorFromJsonDepth(tokenizer, arr_content[cursor..obj_end], depth + 1);
                    cursor = obj_end;
                }
            }
        }
        return;
    }

    // Set add_special for known post-processor types
    if (std.mem.eql(u8, type_str, "BertProcessing") or
        std.mem.eql(u8, type_str, "TemplateProcessing"))
    {
        tokenizer.postproc.add_special = 1;
    } else if (std.mem.eql(u8, type_str, "RobertaProcessing")) {
        tokenizer.postproc.add_special = 1;
        tokenizer.postproc.pair = 1;
    } else {
        return;
    }

    // For BertProcessing: parse cls/sep from ["[CLS]", 1] arrays
    // For TemplateProcessing: parse from special_tokens map
    // In both cases, try to extract token strings
    var cls_str: ?[]const u8 = null;
    var sep_str: ?[]const u8 = null;

    if (std.mem.eql(u8, type_str, "BertProcessing")) {
        // BertProcessing stores cls/sep as ["token_string", id] arrays.
        // Try array format first, fall back to plain string.
        cls_str = findJsonFieldArrayString(json_bytes, "\"cls\"") orelse
            findJsonFieldString(json_bytes, "\"cls\"");
        sep_str = findJsonFieldArrayString(json_bytes, "\"sep\"") orelse
            findJsonFieldString(json_bytes, "\"sep\"");
    }

    // For TemplateProcessing: determine CLS/SEP from the "single" template
    // structure, not from special_tokens map iteration order (which is
    // undefined in JSON). SpecialToken before Sequence(A) → CLS; after → SEP.
    if (std.mem.eql(u8, type_str, "TemplateProcessing")) {
        if (findSection(json_bytes, "\"single\"")) |single_raw| {
            // Scope to just the [...] array (findSection returns from '[' to EOF)
            const single_end = findMatchingBrace(single_raw, '[', ']') orelse single_raw.len;
            const single_section = single_raw[0..single_end];
            const seq_pos = std.mem.indexOf(u8, single_section, "\"Sequence\"");
            var search_pos: usize = 0;
            while (std.mem.indexOfPos(u8, single_section, search_pos, "\"SpecialToken\"")) |spec_pos| {
                // Extract the "id" string from the SpecialToken object.
                // Limit search scope to avoid crossing into the next entry.
                const search_end = @min(spec_pos + 200, single_section.len);
                const spec_slice = single_section[spec_pos..search_end];
                if (findJsonFieldString(spec_slice, "\"id\"")) |id_str| {
                    if (seq_pos) |sp| {
                        if (spec_pos < sp) {
                            if (cls_str == null) cls_str = id_str;
                        } else {
                            if (sep_str == null) sep_str = id_str;
                        }
                    } else {
                        // No Sequence in template — treat first token as CLS
                        if (cls_str == null) cls_str = id_str;
                    }
                }
                search_pos = spec_pos + "\"SpecialToken\"".len;
            }
        }
    }

    // Default token strings (skip for TemplateProcessing — its cls/sep
    // assignment is determined by template ordering, not convention)
    if (cls_str == null and !std.mem.eql(u8, type_str, "TemplateProcessing")) {
        if (std.mem.eql(u8, type_str, "RobertaProcessing")) {
            cls_str = "<s>";
            sep_str = if (sep_str == null) "</s>" else sep_str;
        } else {
            cls_str = "[CLS]";
            sep_str = if (sep_str == null) "[SEP]" else sep_str;
        }
    }
    // For BertProcessing/RobertaProcessing, default sep to cls.
    // For TemplateProcessing, only add SEP if explicitly in special_tokens.
    if (sep_str == null and !std.mem.eql(u8, type_str, "TemplateProcessing")) {
        sep_str = cls_str;
    }

    // Copy token strings into postproc
    if (cls_str) |cls| {
        const copy_len = @min(cls.len, tokenizer.postproc.cls_token.len - 1);
        @memcpy(tokenizer.postproc.cls_token[0..copy_len], cls[0..copy_len]);
        tokenizer.postproc.cls_token[copy_len] = 0;
    }
    if (sep_str) |sep| {
        const copy_len = @min(sep.len, tokenizer.postproc.sep_token.len - 1);
        @memcpy(tokenizer.postproc.sep_token[0..copy_len], sep[0..copy_len]);
        tokenizer.postproc.sep_token[copy_len] = 0;
    }

    // Resolve cls_id/sep_id from added tokens
    if (tokenizer.postproc.cls_id == -1) {
        const cls_z: [*:0]const u8 = @ptrCast(&tokenizer.postproc.cls_token);
        if (tok_fns.tokenizer_added_token_find(tokenizer, cls_z)) |added| {
            tokenizer.postproc.cls_id = added.id;
        }
    }
    if (tokenizer.postproc.sep_id == -1) {
        const sep_z: [*:0]const u8 = @ptrCast(&tokenizer.postproc.sep_token);
        if (tok_fns.tokenizer_added_token_find(tokenizer, sep_z)) |added| {
            tokenizer.postproc.sep_id = added.id;
        }
    }
}

fn applyDecoderFromJson(tokenizer: *ct.Tokenizer, json_bytes: []const u8, arena_allocator: std.mem.Allocator) void {
    var scanner = std.json.Scanner.initCompleteInput(arena_allocator, json_bytes);
    const decoder = parseDecoder(arena_allocator, &scanner) catch return;
    tokenizer.decoder.strip_start = @intCast(decoder.strip_start);
    tokenizer.decoder.strip_stop = @intCast(decoder.strip_stop);
    tokenizer.decoder.wordpiece = if (std.mem.eql(u8, decoder.type, "WordPiece")) 1 else 0;
    tokenizer.decoder.cleanup = if (decoder.cleanup) 1 else 0;
    tokenizer.decoder.add_prefix_space = if (decoder.add_prefix_space) 1 else 0;
    tokenizer.decoder.metaspace = if (decoder.metaspace) 1 else 0;
}

fn apply_root_metadata_without_postprocessor(arena: *std.heap.ArenaAllocator, tokenizer: *ct.Tokenizer, root: schema.TokenizerRoot) !void {
    try apply_added_tokens(tokenizer, root.added_tokens);

    // Apply normalizer settings
    const normalizer_spec = ct.NormalizerSpec{
        .type = if (root.normalizer.type.len > 0) (arena.allocator().dupeZ(u8, root.normalizer.type) catch return error.BuildFailed).ptr else null,
        .lowercase = if (root.normalizer.lowercase) 1 else 0,
        .strip_accents = if (root.normalizer.strip_accents) 1 else 0,
        .nfc = if (root.normalizer.nfc) 1 else 0,
        .nfd = if (root.normalizer.nfd) 1 else 0,
        .nfkc = if (root.normalizer.nfkc) 1 else 0,
        .nfkd = if (root.normalizer.nfkd) 1 else 0,
        .clean_text = if (root.normalizer.clean_text) 1 else 0,
        .handle_chinese_chars = if (root.normalizer.handle_chinese_chars) 1 else 0,
        // SentencePiece-style normalizers
        .prepend = if (root.normalizer.prepend) |p| (std.heap.c_allocator.dupeZ(u8, p) catch return error.BuildFailed).ptr else null,
        .replace_pattern = if (root.normalizer.replace_pattern) |p| (std.heap.c_allocator.dupeZ(u8, p) catch return error.BuildFailed).ptr else null,
        .replace_content = if (root.normalizer.replace_content) |c_val| (std.heap.c_allocator.dupeZ(u8, c_val) catch return error.BuildFailed).ptr else null,
    };
    tok_fns.tokenizer_apply_normalizer_spec(tokenizer, &normalizer_spec);

    // Apply pre_tokenizer settings
    const pretokenizer_spec = ct.PreTokenizerSpec{
        .type = if (root.pre_tokenizer.type.len > 0) (arena.allocator().dupeZ(u8, root.pre_tokenizer.type) catch return error.BuildFailed).ptr else null,
        .add_prefix_space = if (root.pre_tokenizer.add_prefix_space) 1 else 0,
        .trim_offsets = if (root.pre_tokenizer.trim_offsets) 1 else 0,
        .use_regex = if (root.pre_tokenizer.use_regex) 1 else 0,
        .byte_level = if (root.pre_tokenizer.byte_level) 1 else 0,
        .whitespace = if (root.pre_tokenizer.whitespace) 1 else 0,
        .punctuation = if (root.pre_tokenizer.punctuation) 1 else 0,
        .pattern = if (root.pre_tokenizer.pattern) |p| (arena.allocator().dupeZ(u8, p) catch return error.BuildFailed).ptr else null,
        .regex_split = if (root.pre_tokenizer.regex_split) 1 else 0,
        .regex_invert = if (root.pre_tokenizer.regex_invert) 1 else 0,
        .metaspace = if (root.pre_tokenizer.metaspace) 1 else 0,
    };
    tok_fns.tokenizer_apply_pretokenizer_spec(tokenizer, &pretokenizer_spec);
}

fn apply_root_postprocessor_metadata(arena: *std.heap.ArenaAllocator, tokenizer: *ct.Tokenizer, root: schema.TokenizerRoot) !void {
    // Apply post_processor settings from JSON.
    // When JSON explicitly specifies a post_processor, use those settings.
    // When JSON has no post_processor (null), clear any model default (e.g.,
    // WordPiece init sets add_special=1 for BERT-like behavior, but we must
    // respect the JSON configuration as authoritative).
    if (root.post_processor.type.len > 0 or root.post_processor.add_special or root.post_processor.pair or root.post_processor.cls_token != null or root.post_processor.sep_token != null) {
        const postprocessor_spec = ct.PostProcessorSpec{
            .type = if (root.post_processor.type.len > 0) (arena.allocator().dupeZ(u8, root.post_processor.type) catch return error.BuildFailed).ptr else null,
            .add_special = if (root.post_processor.add_special) 1 else 0,
            .pair = if (root.post_processor.pair) 1 else 0,
            .cls_token = if (root.post_processor.cls_token) |cls| (arena.allocator().dupeZ(u8, cls) catch return error.BuildFailed).ptr else null,
            .sep_token = if (root.post_processor.sep_token) |sep| (arena.allocator().dupeZ(u8, sep) catch return error.BuildFailed).ptr else null,
        };
        tok_fns.tokenizer_apply_postprocessor_spec(tokenizer, &postprocessor_spec);
        // Reset IDs so they're re-resolved from the configured token strings.
        // WordPiece init sets cls_id/sep_id from default [CLS]/[SEP] vocab lookup,
        // but BertProcessing may configure different tokens (e.g. <s>/<\/s>).
        tokenizer.postproc.cls_id = -1;
        tokenizer.postproc.sep_id = -1;
    }

    // Resolve cls_id/sep_id from added tokens when post_processor is active.
    // Look up the token strings in the added_tokens list to find the correct IDs.
    if (tokenizer.postproc.add_special != 0 and tokenizer.postproc.cls_id == -1) {
        const cls_z: [*:0]const u8 = @ptrCast(&tokenizer.postproc.cls_token);
        if (tok_fns.tokenizer_added_token_find(tokenizer, cls_z)) |added| {
            tokenizer.postproc.cls_id = added.id;
        }
    }
    if (tokenizer.postproc.add_special != 0 and tokenizer.postproc.sep_id == -1) {
        const sep_z: [*:0]const u8 = @ptrCast(&tokenizer.postproc.sep_token);
        if (tok_fns.tokenizer_added_token_find(tokenizer, sep_z)) |added| {
            tokenizer.postproc.sep_id = added.id;
        }
    }
}

fn apply_root_decoder_metadata(tokenizer: *ct.Tokenizer, root: schema.TokenizerRoot) void {
    // Apply decoder settings (e.g., Strip decoder for SentencePiece)
    tokenizer.decoder.strip_start = @intCast(root.decoder.strip_start);
    tokenizer.decoder.strip_stop = @intCast(root.decoder.strip_stop);
    tokenizer.decoder.wordpiece = if (std.mem.eql(u8, root.decoder.type, "WordPiece")) 1 else 0;
    tokenizer.decoder.cleanup = if (root.decoder.cleanup) 1 else 0;
    tokenizer.decoder.add_prefix_space = if (root.decoder.add_prefix_space) 1 else 0;
    tokenizer.decoder.metaspace = if (root.decoder.metaspace) 1 else 0;
}

fn apply_root_metadata(arena: *std.heap.ArenaAllocator, tokenizer: *ct.Tokenizer, root: schema.TokenizerRoot) !void {
    try apply_root_metadata_without_postprocessor(arena, tokenizer, root);
    try apply_root_postprocessor_metadata(arena, tokenizer, root);
    apply_root_decoder_metadata(tokenizer, root);
}

fn build_tokenizer_from_root(arena: *std.heap.ArenaAllocator, root: schema.TokenizerRoot) !*ct.Tokenizer {
    const model_type_name = root.model.type;
    if (std.mem.eql(u8, model_type_name, "BPE")) {
        const tokenizer = try build_bpe(arena, root.model);
        try apply_root_metadata(arena, tokenizer, root);
        return tokenizer;
    } else if (std.mem.eql(u8, model_type_name, "WordPiece")) {
        const tokenizer = try build_wordpiece(arena, root.model);
        try apply_root_metadata(arena, tokenizer, root);
        return tokenizer;
    } else if (std.mem.eql(u8, model_type_name, "Unigram")) {
        const tokenizer = try build_unigram(arena, root.model);
        try apply_root_metadata(arena, tokenizer, root);
        return tokenizer;
    } else {
        return error.UnsupportedModel;
    }
}

fn build_bpe(arena: *std.heap.ArenaAllocator, model: schema.Model) !*ct.Tokenizer {
    var t_start: i128 = std.time.nanoTimestamp();

    const vocab_len = model.vocab.len;
    const vocab_arr = try arena.allocator().alloc(ct.TokenIdPair, vocab_len);
    for (model.vocab, 0..) |entry, entry_idx| {
        const token_dup = try arena.allocator().dupeZ(u8, entry.token);
        vocab_arr[entry_idx] = .{ .token = token_dup.ptr, .id = entry.id };
    }
    {
        const now = std.time.nanoTimestamp();
        const duration_ms = @as(f64, @floatFromInt(now - t_start)) / 1_000_000.0;
        log.trace("tokenizer", "Vocab copy", .{ .duration_ms = duration_ms }, @src());
        t_start = now;
    }

    const merges_len = if (model.merges) |m| m.len else 0;
    const merges_arr = try arena.allocator().alloc(ct.BpeMergePair, merges_len);
    if (model.merges) |m| {
        for (m, 0..) |merge_str, merge_idx| {
            const parts = std.mem.splitScalar(u8, merge_str, ' ');
            var part_iter = parts;
            const left = part_iter.next() orelse merge_str;
            const right = part_iter.next() orelse "";
            const left_dup = try arena.allocator().dupeZ(u8, left);
            const right_dup = try arena.allocator().dupeZ(u8, right);
            merges_arr[merge_idx] = .{ .a = left_dup.ptr, .b = right_dup.ptr };
        }
    }
    {
        const now = std.time.nanoTimestamp();
        const duration_ms = @as(f64, @floatFromInt(now - t_start)) / 1_000_000.0;
        log.trace("tokenizer", "Merges copy", .{ .duration_ms = duration_ms }, @src());
        t_start = now;
    }

    const model_spec = ct.BpeModelSpec{
        .vocab = vocab_arr.ptr,
        .vocab_len = vocab_len,
        .merges = merges_arr.ptr,
        .merges_len = merges_len,
        .unk_token = if (model.unk_token) |u| (try arena.allocator().dupeZ(u8, u)).ptr else null,
    };
    const result = bpe.tokenizer_bpe_create_from_spec(@ptrCast(&model_spec)) orelse return error.BuildFailed;
    {
        const now = std.time.nanoTimestamp();
        const duration_ms = @as(f64, @floatFromInt(now - t_start)) / 1_000_000.0;
        log.trace("tokenizer", "BPE create", .{ .duration_ms = duration_ms }, @src());
    }
    return result;
}

fn build_wordpiece(arena: *std.heap.ArenaAllocator, model: schema.Model) !*ct.Tokenizer {
    const vocab_len = model.vocab.len;
    const vocab_arr = try arena.allocator().alloc(ct.TokenIdPair, vocab_len);
    for (model.vocab, 0..) |entry, entry_idx| {
        const token_dup = try arena.allocator().dupeZ(u8, entry.token);
        vocab_arr[entry_idx] = .{ .token = token_dup.ptr, .id = entry.id };
    }
    const model_spec = ct.WordPieceModelSpec{
        .vocab = vocab_arr.ptr,
        .vocab_len = vocab_len,
        .unk_token = if (model.unk_token) |u| (try arena.allocator().dupeZ(u8, u)).ptr else null,
        .max_input_chars_per_word = model.max_input_chars_per_word,
    };
    return wordpiece_model.tokenizer_wordpiece_create_from_spec(@ptrCast(&model_spec)) orelse error.BuildFailed;
}

fn build_unigram(arena: *std.heap.ArenaAllocator, model: schema.Model) !*ct.Tokenizer {
    const vocab_len = model.vocab.len;
    const vocab_arr = try arena.allocator().alloc(ct.UnigramVocabEntry, vocab_len);
    for (model.vocab, 0..) |entry, entry_idx| {
        const token_dup = try arena.allocator().dupeZ(u8, entry.token);
        vocab_arr[entry_idx] = .{ .token = token_dup.ptr, .score = entry.score, .id = entry.id };
    }
    const model_spec = ct.UnigramModelSpec{
        .vocab = vocab_arr.ptr,
        .vocab_len = vocab_len,
        .unk_token = if (model.unk_token) |u| (try arena.allocator().dupeZ(u8, u)).ptr else null,
        .bos_token = if (model.bos_token) |u| (try arena.allocator().dupeZ(u8, u)).ptr else null,
        .eos_token = if (model.eos_token) |u| (try arena.allocator().dupeZ(u8, u)).ptr else null,
    };
    return unigram_model.tokenizer_unigram_create_from_spec(@ptrCast(&model_spec)) orelse error.BuildFailed;
}

fn apply_added_tokens(tokenizer: *ct.Tokenizer, added_tokens: []const schema.AddedToken) !void {
    for (added_tokens) |token_entry| {
        const content_dup = try std.heap.c_allocator.dupeZ(u8, token_entry.content);
        defer std.heap.c_allocator.free(content_dup);
        const added_node = tok_fns.tokenizer_added_token_add(tokenizer, content_dup.ptr, token_entry.id, if (token_entry.special) 1 else 0);
        if (added_node == null) return error.BuildFailed;
        added_node.?.single_word = if (token_entry.single_word) 1 else 0;
        added_node.?.lstrip = if (token_entry.lstrip) 1 else 0;
        added_node.?.rstrip = if (token_entry.rstrip) 1 else 0;
        added_node.?.normalized = if (token_entry.normalized) 1 else 0;
    }
}

// =============================================================================
// Tests
// =============================================================================

// Note: Most loader functions (load_from_slice_streaming, tokenizer_loader_from_dir, etc.)
// require file I/O and full JSON parsing. They are tested via integration tests.
// Helper functions like detectModelType, findJsonFieldValue, and unescapeJsonStringFast
// can be unit-tested.

test "detectModelType finds BPE type" {
    const json =
        \\{"model": {"type": "BPE", "vocab": {}}}
    ;
    const model_type = detectModelType(json);
    try std.testing.expectEqualStrings("BPE", model_type);
}

test "load_from_slice_streaming accepts valid minimal bpe" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] },
        \\  "added_tokens": [],
        \\  "normalizer": null,
        \\  "pre_tokenizer": null,
        \\  "post_processor": null,
        \\  "decoder": null
        \\}
    ;

    const root = try load_from_slice_streaming(std.testing.allocator, json);
    try std.testing.expectEqualStrings("BPE", root.model.type);
    try std.testing.expectEqual(@as(usize, 1), root.model.vocab.len);
    try std.testing.expectEqual(@as(usize, 0), root.model.merges.?.len);
}

test "findJsonFieldValue extracts number" {
    const json =
        \\{"id": 123, "name": "test"}
    ;
    const value = findJsonFieldValue(json, "\"id\"");
    try std.testing.expect(value != null);
    try std.testing.expectEqualStrings("123", value.?);
}

test "unescapeJsonStringFast handles escape sequences" {
    const allocator = std.testing.allocator;
    const input = "hello\\nworld";
    const result = try unescapeJsonStringFast(allocator, input);
    defer allocator.free(result);
    try std.testing.expectEqualStrings("hello\nworld", result);
}

test "unescapeJsonStringFast handles multiple escape types" {
    const allocator = std.testing.allocator;

    // Test \t, \r, \n
    {
        const input = "line1\\tline2\\rline3\\nline4";
        const result = try unescapeJsonStringFast(allocator, input);
        defer allocator.free(result);
        try std.testing.expectEqualStrings("line1\tline2\rline3\nline4", result);
    }

    // Test \\, \", \/
    {
        const input = "path\\\\to\\\"file\\\"/";
        const result = try unescapeJsonStringFast(allocator, input);
        defer allocator.free(result);
        try std.testing.expectEqualStrings("path\\to\"file\"/", result);
    }

    // Test \b (backspace) and \f (form feed)
    {
        const input = "text\\bwith\\fspecial";
        const result = try unescapeJsonStringFast(allocator, input);
        defer allocator.free(result);
        try std.testing.expectEqualStrings("text\x08with\x0Cspecial", result);
    }
}

test "unescapeJsonStringFast handles empty string" {
    const allocator = std.testing.allocator;
    const input = "";
    const result = try unescapeJsonStringFast(allocator, input);
    defer allocator.free(result);
    try std.testing.expectEqualStrings("", result);
}

test "unescapeJsonStringFast handles string without escapes" {
    const allocator = std.testing.allocator;
    const input = "hello world";
    const result = try unescapeJsonStringFast(allocator, input);
    defer allocator.free(result);
    try std.testing.expectEqualStrings("hello world", result);
}

test "detectModelType finds WordPiece type" {
    const json =
        \\{"model": {"type": "WordPiece", "vocab": {}}}
    ;
    const model_type = detectModelType(json);
    try std.testing.expectEqualStrings("WordPiece", model_type);
}

test "detectModelType finds Unigram type" {
    const json =
        \\{"model": {"type": "Unigram", "vocab": []}}
    ;
    const model_type = detectModelType(json);
    try std.testing.expectEqualStrings("Unigram", model_type);
}

test "detectModelType defaults to BPE when not found" {
    const json =
        \\{"normalizer": {"type": "NFC"}}
    ;
    const model_type = detectModelType(json);
    try std.testing.expectEqualStrings("BPE", model_type);
}

test "detectModelType handles malformed JSON" {
    const json = "not valid json at all";
    const model_type = detectModelType(json);
    try std.testing.expectEqualStrings("BPE", model_type);
}

test "findJsonFieldValue extracts string value" {
    const json =
        \\{"name": "test", "count": 42}
    ;
    const value = findJsonFieldValue(json, "\"name\"");
    try std.testing.expect(value != null);
    try std.testing.expectEqualStrings("\"test\"", value.?);
}

test "findJsonFieldValue extracts boolean true" {
    const json =
        \\{"enabled": true, "other": false}
    ;
    const value = findJsonFieldValue(json, "\"enabled\"");
    try std.testing.expect(value != null);
    try std.testing.expectEqualStrings("true", value.?);
}

test "findJsonFieldValue extracts boolean false" {
    const json =
        \\{"enabled": false}
    ;
    const value = findJsonFieldValue(json, "\"enabled\"");
    try std.testing.expect(value != null);
    try std.testing.expectEqualStrings("false", value.?);
}

test "findJsonFieldValue returns null for missing field" {
    const json =
        \\{"name": "test"}
    ;
    const value = findJsonFieldValue(json, "\"missing\"");
    try std.testing.expect(value == null);
}

test "findJsonFieldValue handles fields with whitespace" {
    const json =
        \\{"name"  :   123  ,  "other": 456}
    ;
    const value = findJsonFieldValue(json, "\"name\"");
    try std.testing.expect(value != null);
    try std.testing.expectEqualStrings("123", value.?);
}

test "findJsonFieldString extracts quoted string" {
    const json =
        \\{"type": "BPE", "name": "tokenizer"}
    ;
    const value = findJsonFieldString(json, "\"type\"");
    try std.testing.expect(value != null);
    try std.testing.expectEqualStrings("BPE", value.?);
}

test "findJsonFieldString extracts value without outer quotes" {
    const json =
        \\{"path": "test_value"}
    ;
    const value = findJsonFieldString(json, "\"path\"");
    try std.testing.expect(value != null);
    try std.testing.expectEqualStrings("test_value", value.?);
}

test "findJsonFieldString returns null for non-string value" {
    const json =
        \\{"count": 42}
    ;
    const value = findJsonFieldString(json, "\"count\"");
    try std.testing.expect(value == null);
}

test "findJsonFieldString returns null for missing field" {
    const json =
        \\{"name": "test"}
    ;
    const value = findJsonFieldString(json, "\"missing\"");
    try std.testing.expect(value == null);
}

test "findQuotedString finds first quoted string" {
    const input =
        \\some text "hello world" more text
    ;
    const result = findQuotedString(input);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("hello world", result.?);
}

test "findQuotedString returns null when no quotes" {
    const input = "no quotes here";
    const result = findQuotedString(input);
    try std.testing.expect(result == null);
}

test "findQuotedString handles unclosed quote" {
    // findQuotedString returns content from opening quote to end of string
    // when there's no closing quote (it doesn't validate)
    const input = "some text \"unclosed";
    const result = findQuotedString(input);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("unclosed", result.?);
}

test "parseVocabFastSection parses simple vocab" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const json =
        \\{"hello": 0, "world": 1, "test": 2}
    ;
    var vocab_entries = ManagedArrayList(schema.TokenId).init(allocator);

    try parseVocabFastSection(allocator, json, &vocab_entries);

    try std.testing.expectEqual(@as(usize, 3), vocab_entries.items.len);
    try std.testing.expectEqualStrings("hello", vocab_entries.items[0].token);
    try std.testing.expectEqual(@as(i32, 0), vocab_entries.items[0].id);
    try std.testing.expectEqualStrings("world", vocab_entries.items[1].token);
    try std.testing.expectEqual(@as(i32, 1), vocab_entries.items[1].id);
}

test "parseVocabFastSection handles escaped tokens" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // JSON with escape sequences: backslash followed by 'n' should become newline
    // Use string concatenation to get actual backslash-n in the JSON
    const json = "{\"hello" ++ [_]u8{ '\\', 'n' } ++ "world\": 0}";
    var vocab_entries = ManagedArrayList(schema.TokenId).init(allocator);

    try parseVocabFastSection(allocator, json, &vocab_entries);

    try std.testing.expectEqual(@as(usize, 1), vocab_entries.items.len);
    // The function unescapes \n to newline
    try std.testing.expectEqualStrings("hello\nworld", vocab_entries.items[0].token);
}

test "parseVocabFastSection skips non-numeric values" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const json =
        \\{"token": 0, "config": {"nested": "value"}, "other": 1}
    ;
    var vocab_entries = ManagedArrayList(schema.TokenId).init(allocator);

    try parseVocabFastSection(allocator, json, &vocab_entries);

    try std.testing.expectEqual(@as(usize, 2), vocab_entries.items.len);
    try std.testing.expectEqualStrings("token", vocab_entries.items[0].token);
    try std.testing.expectEqualStrings("other", vocab_entries.items[1].token);
}

test "parseVocabFastSection rejects sparse ids outside i32 range" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const json =
        \\{"<unk>": 0, "boom": 2147483648}
    ;
    var vocab_entries = ManagedArrayList(schema.TokenId).init(allocator);

    try std.testing.expectError(error.InvalidVocab, parseVocabFastSection(allocator, json, &vocab_entries));
}

test "parseMergesFastSection parses array format" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const json =
        \\[["a", "b"], ["c", "d"], ["e", "f"]]
    ;
    var merge_entries = ManagedArrayList([]const u8).init(allocator);

    try parseMergesFastSection(allocator, json, &merge_entries);

    try std.testing.expectEqual(@as(usize, 3), merge_entries.items.len);
    try std.testing.expectEqualStrings("a b", merge_entries.items[0]);
    try std.testing.expectEqualStrings("c d", merge_entries.items[1]);
    try std.testing.expectEqualStrings("e f", merge_entries.items[2]);
}

test "parseMergesFastSection parses mixed array format" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Real-world format: array of arrays
    const json =
        \\[["a", "b"], ["c", "d"]]
    ;
    var merge_entries = ManagedArrayList([]const u8).init(allocator);

    try parseMergesFastSection(allocator, json, &merge_entries);

    try std.testing.expectEqual(@as(usize, 2), merge_entries.items.len);
    try std.testing.expectEqualStrings("a b", merge_entries.items[0]);
    try std.testing.expectEqualStrings("c d", merge_entries.items[1]);
}

test "parseNormalizer parses NFC normalizer" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{"type": "NFC"}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const normalizer = try parseNormalizer(arena.allocator(), &scanner);

    try std.testing.expect(normalizer.nfc);
    try std.testing.expect(!normalizer.nfd);
    try std.testing.expect(!normalizer.nfkc);
}

test "parseNormalizer parses NFKD normalizer" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{"type": "NFKD"}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const normalizer = try parseNormalizer(arena.allocator(), &scanner);

    try std.testing.expect(normalizer.nfkd);
    try std.testing.expect(!normalizer.nfc);
    try std.testing.expect(!normalizer.nfd);
    try std.testing.expect(!normalizer.nfkc);
}

test "parseNormalizer parses lowercase normalizer" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{"type": "Lowercase"}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const normalizer = try parseNormalizer(arena.allocator(), &scanner);

    try std.testing.expect(normalizer.lowercase);
}

test "parseNormalizer handles explicit flags" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{"type": "Custom", "lowercase": true, "strip_accents": true, "nfc": true}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const normalizer = try parseNormalizer(arena.allocator(), &scanner);

    try std.testing.expect(normalizer.lowercase);
    try std.testing.expect(normalizer.strip_accents);
    try std.testing.expect(normalizer.nfc);
}

test "parseNormalizer preserves explicit BertNormalizer false flags" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{"type": "BertNormalizer", "clean_text": false, "handle_chinese_chars": false, "strip_accents": false, "lowercase": false}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const normalizer = try parseNormalizer(arena.allocator(), &scanner);

    try std.testing.expect(!normalizer.clean_text);
    try std.testing.expect(!normalizer.handle_chinese_chars);
    try std.testing.expect(!normalizer.strip_accents);
    try std.testing.expect(!normalizer.lowercase);
}

test "parseNormalizer rejects unknown type" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{"type": "DoesNotExist"}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    try std.testing.expectError(error.InvalidNormalizer, parseNormalizer(arena.allocator(), &scanner));
}

test "parseNormalizer rejects Replace without pattern" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{"type": "Replace", "content": "x"}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    try std.testing.expectError(error.InvalidNormalizer, parseNormalizer(arena.allocator(), &scanner));
}

test "parseNormalizer rejects Prepend without prepend text" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{"type": "Prepend"}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    try std.testing.expectError(error.InvalidNormalizer, parseNormalizer(arena.allocator(), &scanner));
}

test "parseNormalizer handles null normalizer" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json = "null";

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const normalizer = try parseNormalizer(arena.allocator(), &scanner);

    try std.testing.expect(!normalizer.lowercase);
    try std.testing.expect(!normalizer.nfc);
}

test "parsePreTokenizer parses ByteLevel type" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{"type": "ByteLevel", "add_prefix_space": true}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const pretok = try parsePreTokenizer(arena.allocator(), &scanner);

    try std.testing.expect(pretok.byte_level);
    try std.testing.expect(pretok.add_prefix_space);
}

test "parsePreTokenizer parses Whitespace type" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{"type": "Whitespace"}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const pretok = try parsePreTokenizer(arena.allocator(), &scanner);

    try std.testing.expect(pretok.whitespace);
}

test "parsePreTokenizer handles null pretokenizer" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json = "null";

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const pretok = try parsePreTokenizer(arena.allocator(), &scanner);

    try std.testing.expect(!pretok.byte_level);
    try std.testing.expect(!pretok.whitespace);
}

test "parsePostProcessor parses BertProcessing type" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{"type": "BertProcessing"}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const postproc = try parsePostProcessor(arena.allocator(), &scanner);

    try std.testing.expect(postproc.add_special);
    try std.testing.expectEqualStrings("[CLS]", postproc.cls_token.?);
    try std.testing.expectEqualStrings("[SEP]", postproc.sep_token.?);
}

test "parsePostProcessor parses RobertaProcessing type" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{"type": "RobertaProcessing"}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const postproc = try parsePostProcessor(arena.allocator(), &scanner);

    try std.testing.expect(postproc.add_special);
    try std.testing.expect(postproc.pair);
    try std.testing.expectEqualStrings("<s>", postproc.cls_token.?);
    try std.testing.expectEqualStrings("</s>", postproc.sep_token.?);
}

test "parsePostProcessor rejects TemplateProcessing with undefined special token mapping" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{
        \\  "type": "TemplateProcessing",
        \\  "single": [
        \\    {"SpecialToken": {"id": "<s>", "type_id": 0}},
        \\    {"Sequence": {"id": "A", "type_id": 0}}
        \\  ],
        \\  "pair": [
        \\    {"SpecialToken": {"id": "<s>", "type_id": 0}},
        \\    {"Sequence": {"id": "A", "type_id": 0}},
        \\    {"Sequence": {"id": "B", "type_id": 1}}
        \\  ],
        \\  "special_tokens": {}
        \\}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    try std.testing.expectError(error.InvalidPostProcessor, parsePostProcessor(arena.allocator(), &scanner));
}

test "parsePostProcessor accepts Sequence wrapping ByteLevel and TemplateProcessing" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{
        \\  "type": "Sequence",
        \\  "processors": [
        \\    {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true},
        \\    {
        \\      "type": "TemplateProcessing",
        \\      "single": [
        \\        {"SpecialToken": {"id": "<s>", "type_id": 0}},
        \\        {"Sequence": {"id": "A", "type_id": 0}}
        \\      ],
        \\      "pair": [
        \\        {"SpecialToken": {"id": "<s>", "type_id": 0}},
        \\        {"Sequence": {"id": "A", "type_id": 0}},
        \\        {"Sequence": {"id": "B", "type_id": 1}}
        \\      ],
        \\      "special_tokens": {
        \\        "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}
        \\      }
        \\    }
        \\  ]
        \\}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const postproc = try parsePostProcessor(arena.allocator(), &scanner);

    try std.testing.expectEqualStrings("TemplateProcessing", postproc.type);
    try std.testing.expect(postproc.add_special);
    try std.testing.expectEqualStrings("<s>", postproc.cls_token.?);
    try std.testing.expect(postproc.sep_token == null);
}

test "parsePostProcessor infers TemplateProcessing BOS and EOS from single template order" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{
        \\  "type": "TemplateProcessing",
        \\  "single": [
        \\    {"SpecialToken": {"id": "<s>", "type_id": 0}},
        \\    {"Sequence": {"id": "A", "type_id": 0}},
        \\    {"SpecialToken": {"id": "</s>", "type_id": 0}}
        \\  ],
        \\  "pair": [
        \\    {"SpecialToken": {"id": "<s>", "type_id": 0}},
        \\    {"Sequence": {"id": "A", "type_id": 0}},
        \\    {"SpecialToken": {"id": "</s>", "type_id": 0}},
        \\    {"Sequence": {"id": "B", "type_id": 1}},
        \\    {"SpecialToken": {"id": "</s>", "type_id": 0}}
        \\  ],
        \\  "special_tokens": {
        \\    "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]},
        \\    "</s>": {"id": "</s>", "ids": [2], "tokens": ["</s>"]}
        \\  }
        \\}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const postproc = try parsePostProcessor(arena.allocator(), &scanner);

    try std.testing.expect(postproc.add_special);
    try std.testing.expectEqualStrings("<s>", postproc.cls_token.?);
    try std.testing.expectEqualStrings("</s>", postproc.sep_token.?);
}

test "parsePostProcessor infers EOS-only TemplateProcessing without default CLS" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{
        \\  "type": "TemplateProcessing",
        \\  "single": [
        \\    {"Sequence": {"id": "A", "type_id": 0}},
        \\    {"SpecialToken": {"id": "<|endoftext|>", "type_id": 0}}
        \\  ],
        \\  "pair": [
        \\    {"Sequence": {"id": "A", "type_id": 0}},
        \\    {"Sequence": {"id": "B", "type_id": 1}},
        \\    {"SpecialToken": {"id": "<|endoftext|>", "type_id": 0}}
        \\  ],
        \\  "special_tokens": {
        \\    "<|endoftext|>": {"id": "<|endoftext|>", "ids": [1], "tokens": ["<|endoftext|>"]}
        \\  }
        \\}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const postproc = try parsePostProcessor(arena.allocator(), &scanner);

    try std.testing.expect(postproc.add_special);
    try std.testing.expect(postproc.cls_token == null);
    try std.testing.expectEqualStrings("<|endoftext|>", postproc.sep_token.?);
}

test "parsePostProcessor handles null postprocessor" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json = "null";

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const postproc = try parsePostProcessor(arena.allocator(), &scanner);

    try std.testing.expect(!postproc.add_special);
    try std.testing.expect(postproc.cls_token == null);
}

test "parseDecoder parses Strip decoder" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{"type": "Strip", "start": 1, "stop": 0}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const decoder = try parseDecoder(arena.allocator(), &scanner);

    try std.testing.expectEqual(@as(i32, 1), decoder.strip_start);
    try std.testing.expectEqual(@as(i32, 0), decoder.strip_stop);
}

test "parseDecoder handles null decoder" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json = "null";

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const decoder = try parseDecoder(arena.allocator(), &scanner);

    try std.testing.expectEqual(@as(i32, 0), decoder.strip_start);
    try std.testing.expectEqual(@as(i32, 0), decoder.strip_stop);
}

test "parseDecoder preserves WordPiece cleanup=false" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{"type": "WordPiece", "prefix": "##", "cleanup": false}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const decoder = try parseDecoder(arena.allocator(), &scanner);

    try std.testing.expectEqualStrings("WordPiece", decoder.type);
    try std.testing.expect(!decoder.cleanup);
}

test "parseDecoder handles ByteFallback decoder" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{
        \\  "type": "ByteFallback"
        \\}
    ;
    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const decoder = try parseDecoder(arena.allocator(), &scanner);
    try std.testing.expectEqualStrings("ByteFallback", decoder.type);
}

test "parseDecoder Sequence maps Replace metaspace and Strip" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{
        \\  "type": "Sequence",
        \\  "decoders": [
        \\    {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
        \\    {"type": "ByteFallback"},
        \\    {"type": "Fuse"},
        \\    {"type": "Strip", "content": " ", "start": 1, "stop": 0}
        \\  ]
        \\}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const decoder = try parseDecoder(arena.allocator(), &scanner);

    try std.testing.expectEqualStrings("Sequence", decoder.type);
    try std.testing.expect(decoder.metaspace);
    try std.testing.expectEqual(@as(i32, 1), decoder.strip_start);
    try std.testing.expectEqual(@as(i32, 0), decoder.strip_stop);
}

test "parseDecoder Sequence preserves nested Metaspace add_prefix_space" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{
        \\  "type": "Sequence",
        \\  "decoders": [
        \\    {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first"}
        \\  ]
        \\}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const decoder = try parseDecoder(arena.allocator(), &scanner);

    try std.testing.expectEqualStrings("Sequence", decoder.type);
    try std.testing.expect(decoder.metaspace);
    try std.testing.expect(decoder.add_prefix_space);
}

test "validateBpeMergeReferences accepts chained merge intermediates" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const vocab = [_]schema.TokenId{
        .{ .token = "h", .id = 0, .score = -1.0 },
        .{ .token = "e", .id = 1, .score = -1.0 },
        .{ .token = "l", .id = 2, .score = -1.0 },
        .{ .token = "o", .id = 3, .score = -1.0 },
    };
    const merges = [_][]const u8{
        "h e",
        "he l",
        "hel l",
        "hell o",
    };

    try validateBpeMergeReferences(arena.allocator(), "BPE", &vocab, &merges);
}

fn appendNestedSequenceSection(
    allocator: std.mem.Allocator,
    field_name: []const u8,
    entry_name: []const u8,
    depth: usize,
    leaf: []const u8,
) ![]u8 {
    var current = try allocator.dupe(u8, leaf);
    errdefer allocator.free(current);
    for (0..depth) |_| {
        const next = try std.fmt.allocPrint(
            allocator,
            "{{\"type\":\"Sequence\",\"{s}\":[{s}]}}",
            .{ entry_name, current },
        );
        allocator.free(current);
        current = next;
    }
    errdefer allocator.free(current);
    const normalizer = if (std.mem.eql(u8, field_name, "normalizer")) current else "null";
    const pre_tokenizer = if (std.mem.eql(u8, field_name, "pre_tokenizer")) current else "null";
    const post_processor = if (std.mem.eql(u8, field_name, "post_processor")) current else "null";
    const decoder = if (std.mem.eql(u8, field_name, "decoder")) current else "null";
    return std.fmt.allocPrint(
        allocator,
        "{{\n  \"version\": \"1.0\",\n  \"model\": {{\"type\": \"BPE\", \"vocab\": {{\"<unk>\": 0, \"a\": 1}}, \"merges\": []}},\n  \"added_tokens\": [{{\"id\": 0, \"content\": \"<unk>\", \"special\": true}}],\n  \"normalizer\": {s},\n  \"pre_tokenizer\": {s},\n  \"post_processor\": {s},\n  \"decoder\": {s}\n}}",
        .{ normalizer, pre_tokenizer, post_processor, decoder },
    );
}

test "load_from_slice_streaming accepts normalizer nesting at depth limit" {
    const json = try appendNestedSequenceSection(
        std.testing.allocator,
        "normalizer",
        "normalizers",
        MAX_JSON_PIPELINE_DEPTH,
        "{\"type\":\"Lowercase\"}",
    );
    defer std.testing.allocator.free(json);

    const root = try load_from_slice_streaming(std.testing.allocator, json);
    try std.testing.expect(root.normalizer.lowercase);
}

test "load_from_slice_streaming rejects normalizer nesting above depth limit" {
    const json = try appendNestedSequenceSection(
        std.testing.allocator,
        "normalizer",
        "normalizers",
        MAX_JSON_PIPELINE_DEPTH + 1,
        "{\"type\":\"Lowercase\"}",
    );
    defer std.testing.allocator.free(json);

    try std.testing.expectError(error.InvalidNormalizer, load_from_slice_streaming(std.testing.allocator, json));
}

test "load_from_slice_streaming rejects pretokenizer nesting above depth limit" {
    const json = try appendNestedSequenceSection(
        std.testing.allocator,
        "pre_tokenizer",
        "pretokenizers",
        MAX_JSON_PIPELINE_DEPTH + 1,
        "{\"type\":\"Whitespace\"}",
    );
    defer std.testing.allocator.free(json);

    try std.testing.expectError(error.InvalidPreTokenizer, load_from_slice_streaming(std.testing.allocator, json));
}

test "load_from_slice_streaming rejects postprocessor nesting above depth limit" {
    const json = try appendNestedSequenceSection(
        std.testing.allocator,
        "post_processor",
        "processors",
        MAX_JSON_PIPELINE_DEPTH + 1,
        "{\"type\":\"ByteLevel\"}",
    );
    defer std.testing.allocator.free(json);

    try std.testing.expectError(error.InvalidPostProcessor, load_from_slice_streaming(std.testing.allocator, json));
}

test "load_from_slice_streaming accepts postprocessor nesting at depth limit" {
    const json = try appendNestedSequenceSection(
        std.testing.allocator,
        "post_processor",
        "processors",
        MAX_JSON_PIPELINE_DEPTH,
        "{\"type\":\"ByteLevel\"}",
    );
    defer std.testing.allocator.free(json);

    const root = try load_from_slice_streaming(std.testing.allocator, json);
    try std.testing.expect(root.post_processor.type.len > 0);
}

test "load_from_slice_streaming rejects decoder nesting above depth limit" {
    const json = try appendNestedSequenceSection(
        std.testing.allocator,
        "decoder",
        "decoders",
        MAX_JSON_PIPELINE_DEPTH + 1,
        "{\"type\":\"ByteLevel\"}",
    );
    defer std.testing.allocator.free(json);

    try std.testing.expectError(error.InvalidDecoder, load_from_slice_streaming(std.testing.allocator, json));
}

test "load_from_slice_streaming accepts decoder nesting at depth limit" {
    const json = try appendNestedSequenceSection(
        std.testing.allocator,
        "decoder",
        "decoders",
        MAX_JSON_PIPELINE_DEPTH,
        "{\"type\":\"ByteLevel\"}",
    );
    defer std.testing.allocator.free(json);

    const root = try load_from_slice_streaming(std.testing.allocator, json);
    try std.testing.expectEqualStrings("Sequence", root.decoder.type);
}

test "load_from_slice_streaming accepts direct ByteFallback decoder" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "model": {"type": "BPE", "vocab": {"<unk>": 0}, "merges": []},
        \\  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
        \\  "normalizer": null,
        \\  "pre_tokenizer": null,
        \\  "post_processor": null,
        \\  "decoder": {"type": "ByteFallback"}
        \\}
    ;

    const root = try load_from_slice_streaming(std.testing.allocator, json);
    try std.testing.expectEqualStrings("ByteFallback", root.decoder.type);
}

test "load_from_slice_streaming requires integration testing" {
    // This function requires:
    // - Full JSON tokenizer.json content
    // - Complex vocab and merges parsing
    // - Arena allocator with proper lifetime management
    // - Metadata section parsing (normalizer, pretokenizer, etc.)
    // Integration tests: tests/tokenizer/test_*.py
}

test "tokenizer_loader_from_dir requires integration testing" {
    // This function requires:
    // - File I/O access to tokenizer.json
    // - Cache directory layout support (snapshots)
    // - Full tokenizer initialization (BPE/WordPiece/Unigram)
    // - Applied configuration (normalizer, pretokenizer, decoder)
    // Integration tests: tests/tokenizer/test_*.py
}

test "tokenizer_loader_from_json_string applies whitespace pretokenizer on lazy bpe path" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "version": "1.0",
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {
        \\      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
        \\      "a": 4, "b": 5
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
        \\  "pre_tokenizer": {"type": "Whitespace"},
        \\  "post_processor": null,
        \\  "decoder": null
        \\}
    ;
    const json_z = try allocator.dupeZ(u8, json);
    defer allocator.free(json_z);

    const tokenizer = tokenizer_loader_from_json_string(json_z.ptr) orelse return error.OutOfMemory;
    defer {
        tokenizer.destroy();
        std.heap.c_allocator.destroy(tokenizer);
    }

    var encoding = std.mem.zeroes(ct.TokenizerEncoding);
    defer @import("encode.zig").tokenizer_encoding_free_struct(&encoding);

    try std.testing.expectEqual(@as(c_int, 0), @import("encode.zig").tokenizer_encode_struct_with_options(tokenizer, "a b", &encoding, .{ .add_special_tokens = false }));
    try std.testing.expectEqual(@as(usize, 2), encoding.ids_len);

    const ids: [*]i32 = @ptrCast(encoding.ids.?);
    try std.testing.expectEqual(@as(i32, 4), ids[0]);
    try std.testing.expectEqual(@as(i32, 5), ids[1]);
}

test "tokenizer_loader_from_json_string preserves sentencepiece normalizer ownership on lazy bpe path" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "version": "1.0",
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {
        \\      "<unk>": 0, "<s>": 1, "</s>": 2,
        \\      "▁": 3,
        \\      "h": 4, "e": 5, "l": 6, "o": 7, "w": 8, "r": 9, "d": 10,
        \\      "▁h": 11, "▁he": 12, "▁hel": 13, "▁hell": 14, "▁hello": 15,
        \\      "▁w": 16, "▁wo": 17, "▁wor": 18, "▁worl": 19, "▁world": 20
        \\    },
        \\    "merges": [
        \\      "▁ h", "▁h e", "▁he l", "▁hel l", "▁hell o",
        \\      "▁ w", "▁w o", "▁wo r", "▁wor l", "▁worl d"
        \\    ]
        \\  },
        \\  "added_tokens": [
        \\    {"id": 0, "content": "<unk>", "special": true},
        \\    {"id": 1, "content": "<s>", "special": true},
        \\    {"id": 2, "content": "</s>", "special": true}
        \\  ],
        \\  "normalizer": {
        \\    "type": "Sequence",
        \\    "normalizers": [
        \\      {"type": "Prepend", "prepend": "▁"},
        \\      {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}
        \\    ]
        \\  },
        \\  "pre_tokenizer": null,
        \\  "post_processor": null,
        \\  "decoder": null
        \\}
    ;
    const json_z = try allocator.dupeZ(u8, json);
    defer allocator.free(json_z);

    const tokenizer = tokenizer_loader_from_json_string(json_z.ptr) orelse return error.OutOfMemory;
    defer {
        tokenizer.destroy();
        std.heap.c_allocator.destroy(tokenizer);
    }

    var encoding = std.mem.zeroes(ct.TokenizerEncoding);
    defer @import("encode.zig").tokenizer_encoding_free_struct(&encoding);

    try std.testing.expectEqual(@as(c_int, 0), @import("encode.zig").tokenizer_encode_struct_with_options(tokenizer, "<s> hello </s> world", &encoding, .{ .add_special_tokens = false }));

    const ids: [*]i32 = @ptrCast(encoding.ids.?);
    try std.testing.expectEqual(@as(usize, 7), encoding.ids_len);
    try std.testing.expectEqualSlices(i32, &.{ 1, 3, 15, 3, 2, 3, 20 }, ids[0..encoding.ids_len]);
}

test "tokenizer_loader_from_json_string accepts empty inert BPE affix defaults" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "version": "1.0",
        \\  "model": {
        \\    "type": "BPE",
        \\    "dropout": null,
        \\    "unk_token": "<unk>",
        \\    "continuing_subword_prefix": "",
        \\    "end_of_word_suffix": "",
        \\    "fuse_unk": false,
        \\    "byte_fallback": false,
        \\    "ignore_merges": false,
        \\    "vocab": { "<unk>": 0, "h": 1, "e": 2, "l": 3, "o": 4 },
        \\    "merges": []
        \\  },
        \\  "added_tokens": [],
        \\  "normalizer": null,
        \\  "pre_tokenizer": null,
        \\  "post_processor": null,
        \\  "decoder": null
        \\}
    ;
    const json_z = try allocator.dupeZ(u8, json);
    defer allocator.free(json_z);

    const tokenizer = tokenizer_loader_from_json_string(json_z.ptr) orelse return error.TestUnexpectedResult;
    defer {
        tokenizer.destroy();
        std.heap.c_allocator.destroy(tokenizer);
    }
}

test "tokenizer_loader_from_json_string requires integration testing" {
    // This function requires:
    // - Complete JSON tokenizer specification
    // - Model type detection and routing
    // - Lazy/full parsing based on model type
    // - Configuration application
    // Integration tests: tests/tokenizer/test_*.py
}
