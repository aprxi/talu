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

    // Fast path: directly scan for vocab and merges sections
    var vocab_entries = ManagedArrayList(schema.TokenId).init(arena_allocator);
    var merge_entries = ManagedArrayList([]const u8).init(arena_allocator);
    var model_type_name: []const u8 = "BPE";

    // Find and parse vocab section directly
    if (findSection(json_content, "\"vocab\"")) |vocab_section| {
        if (vocab_section.len > 0 and vocab_section[0] == '{') {
            // Find matching closing brace
            const vocab_end = findMatchingBrace(vocab_section, '{', '}') orelse vocab_section.len;
            try parseVocabFastSection(arena_allocator, vocab_section[0..vocab_end], &vocab_entries);
        } else if (vocab_section.len > 0 and vocab_section[0] == '[') {
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
                    if (tok != .array_begin) {
                        try scanner.skipValue();
                        continue;
                    }
                    // Inner array: ["token", score]
                    const token_tok = try scanner.nextAlloc(arena_allocator, .alloc_if_needed);
                    const token_str: []const u8 = switch (token_tok) {
                        .string => |s| s,
                        .allocated_string => |s| s,
                        else => {
                            try scanner.skipValue();
                            _ = try scanner.next(); // closing ]
                            continue;
                        },
                    };
                    const score_tok = try scanner.next();
                    const score: f32 = switch (score_tok) {
                        .number => |n| std.fmt.parseFloat(f32, n) catch 0.0,
                        else => 0.0,
                    };
                    _ = try scanner.next(); // closing ]
                    try vocab_entries.append(.{ .token = token_str, .id = next_id, .score = score });
                    next_id += 1;
                }
            }
        }
    }

    // Find and parse merges section directly
    if (findSection(json_content, "\"merges\"")) |merges_section| {
        if (merges_section.len > 0 and merges_section[0] == '[') {
            const merges_end = findMatchingBrace(merges_section, '[', ']') orelse merges_section.len;
            try parseMergesFastSection(arena_allocator, merges_section[0..merges_end], &merge_entries);
        }
    }

    // Find model type using the same method as detectModelType
    // Look within the "model" section for robustness
    model_type_name = detectModelType(json_content);

    // Extract unk_token from model section
    var unk_token_str: ?[]const u8 = null;
    if (findSection(json_content, "\"model\"")) |model_section| {
        const search_len = @min(500, model_section.len);
        const search_region = model_section[0..search_len];
        // Try "unk_token" (string value) — used by WordPiece
        if (std.mem.indexOf(u8, search_region, "\"unk_token\"")) |unk_pos| {
            const after_key = model_section[unk_pos + "\"unk_token\"".len ..];
            if (findQuotedString(after_key)) |val| {
                unk_token_str = val;
            }
        }
        // Try "unk_id" (integer index into vocab) — used by Unigram
        if (unk_token_str == null) {
            if (std.mem.indexOf(u8, search_region, "\"unk_id\"")) |unk_pos| {
                const after_key = model_section[unk_pos + "\"unk_id\"".len ..];
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
                        decoder.* = try parseDecoder(arena_allocator, &scanner);
                    } else {
                        try scanner.skipValue();
                    }
                },
                else => break,
            }
        }
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

/// Fast merges parser - handles both ["a", "b"] and "a b" formats
fn parseMergesFastSection(arena_allocator: std.mem.Allocator, json_bytes: []const u8, merge_entries: *ManagedArrayList([]const u8)) !void {
    // Some models have 500k+ merges, so use a large capacity
    try merge_entries.ensureTotalCapacity(600000);

    var cursor: usize = 0;
    while (cursor < json_bytes.len) {
        // Look for either [ (array format) or " (string format)
        while (cursor < json_bytes.len and json_bytes[cursor] != '[' and json_bytes[cursor] != '"') : (cursor += 1) {}
        if (cursor >= json_bytes.len) break;

        if (json_bytes[cursor] == '[') {
            // Array format: ["a", "b"]
            cursor += 1;

            // Find first string
            while (cursor < json_bytes.len and json_bytes[cursor] != '"') : (cursor += 1) {}
            if (cursor >= json_bytes.len) break;
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

            // Find second string
            while (cursor < json_bytes.len and json_bytes[cursor] != '"') : (cursor += 1) {}
            if (cursor >= json_bytes.len) break;
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

            // Join with space
            const merged = try std.fmt.allocPrint(arena_allocator, "{s} {s}", .{ lhs, rhs });
            merge_entries.appendAssumeCapacity(merged);

            // Skip to end of array
            while (cursor < json_bytes.len and json_bytes[cursor] != ']') : (cursor += 1) {}
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
            if (std.mem.indexOf(u8, merge_str, " ") != null) {
                merge_entries.appendAssumeCapacity(merge_str);
            }
            cursor += 1;
        }
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
        if (ch >= '0' and ch <= '9') {
            // Parse number
            const num_start = cursor;
            while (cursor < json_bytes.len and json_bytes[cursor] >= '0' and json_bytes[cursor] <= '9') : (cursor += 1) {}
            const id_num = std.fmt.parseInt(i32, json_bytes[num_start..cursor], 10) catch continue;

            // Get the key - zero-copy if no escapes
            const key = if (has_escape)
                try unescapeJsonStringFast(arena_allocator, json_bytes[key_start..key_end])
            else
                json_bytes[key_start..key_end];

            vocab_entries.appendAssumeCapacity(.{ .token = key, .id = id_num, .score = -1.0 });
        }
        // else: not a vocab entry, continue scanning
    }
}

fn unescapeJsonStringFast(arena_allocator: std.mem.Allocator, input_bytes: []const u8) ![]const u8 {
    var result = try arena_allocator.alloc(u8, input_bytes.len);
    var out_index: usize = 0;
    var byte_index: usize = 0;
    while (byte_index < input_bytes.len) {
        if (input_bytes[byte_index] == '\\' and byte_index + 1 < input_bytes.len) {
            switch (input_bytes[byte_index + 1]) {
                'n' => result[out_index] = '\n',
                'r' => result[out_index] = '\r',
                't' => result[out_index] = '\t',
                'b' => result[out_index] = 0x08, // backspace
                'f' => result[out_index] = 0x0C, // form feed
                '\\' => result[out_index] = '\\',
                '"' => result[out_index] = '"',
                '/' => result[out_index] = '/',
                else => result[out_index] = input_bytes[byte_index + 1],
            }
            byte_index += 2;
        } else {
            result[out_index] = input_bytes[byte_index];
            byte_index += 1;
        }
        out_index += 1;
    }
    // Shrink allocation to actual size if output is shorter than input
    if (out_index < input_bytes.len) {
        return arena_allocator.realloc(result, out_index) catch result[0..out_index];
    }
    return result;
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
    while (true) {
        const json_token = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
        switch (json_token) {
            .array_end => break,
            .object_begin => {
                var added_entry = schema.AddedToken{
                    .id = 0,
                    .content = "",
                };
                while (true) {
                    const key_token = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    switch (key_token) {
                        .object_end => break,
                        .allocated_string => |key| {
                            if (std.mem.eql(u8, key, "id")) {
                                const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                                added_entry.id = switch (value) {
                                    .allocated_number => |bytes| std.fmt.parseInt(i32, bytes, 10) catch 0,
                                    else => 0,
                                };
                            } else if (std.mem.eql(u8, key, "content")) {
                                const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                                if (value == .allocated_string) added_entry.content = value.allocated_string;
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
                try added_entries.append(added_entry);
            },
            else => return error.InvalidAdded,
        }
    }
}

fn parseNormalizer(arena_allocator: std.mem.Allocator, json_scanner: *std.json.Scanner) !schema.Normalizer {
    var normalizer = schema.Normalizer{};
    const first_token = try json_scanner.next();
    if (first_token == .null) return normalizer;
    if (first_token != .object_begin) return error.InvalidNormalizer;

    while (true) {
        const json_token = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
        switch (json_token) {
            .object_end => break,
            .allocated_string => |key| {
                if (std.mem.eql(u8, key, "type")) {
                    const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    if (value == .allocated_string) normalizer.type = value.allocated_string;
                } else if (std.mem.eql(u8, key, "lowercase")) {
                    const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    normalizer.lowercase = (value == .true);
                } else if (std.mem.eql(u8, key, "strip_accents")) {
                    const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    normalizer.strip_accents = (value == .true);
                } else if (std.mem.eql(u8, key, "nfc") or std.mem.eql(u8, key, "NFC")) {
                    const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    normalizer.nfc = (value == .true);
                } else if (std.mem.eql(u8, key, "nfd") or std.mem.eql(u8, key, "NFD")) {
                    const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    normalizer.nfd = (value == .true);
                } else if (std.mem.eql(u8, key, "nfkc") or std.mem.eql(u8, key, "NFKC")) {
                    const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    normalizer.nfkc = (value == .true);
                } else if (std.mem.eql(u8, key, "clean_text")) {
                    const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    normalizer.clean_text = (value == .true);
                } else if (std.mem.eql(u8, key, "handle_chinese_chars")) {
                    const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    normalizer.handle_chinese_chars = (value == .true);
                } else if (std.mem.eql(u8, key, "prepend")) {
                    // Prepend normalizer: prepend this string to input
                    const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    if (value == .allocated_string) normalizer.prepend = value.allocated_string;
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
                                    if (std.mem.eql(u8, pat_key, "String")) {
                                        const pat_val = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                                        if (pat_val == .allocated_string) normalizer.replace_pattern = pat_val.allocated_string;
                                    } else {
                                        try json_scanner.skipValue();
                                    }
                                },
                                else => {},
                            }
                        }
                    }
                } else if (std.mem.eql(u8, key, "content")) {
                    // Replace normalizer content (replacement string)
                    const value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    if (value == .allocated_string) normalizer.replace_content = value.allocated_string;
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
                        const sub = try parseNormalizer(arena_allocator, json_scanner);
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
        normalizer.clean_text = true;
        normalizer.handle_chinese_chars = true;
        normalizer.lowercase = true;
        normalizer.strip_accents = true;
    } else if (std.mem.eql(u8, normalizer.type, "Lowercase")) {
        normalizer.lowercase = true;
    } else if (std.mem.eql(u8, normalizer.type, "NFC")) {
        normalizer.nfc = true;
    } else if (std.mem.eql(u8, normalizer.type, "NFD")) {
        normalizer.nfd = true;
    } else if (std.mem.eql(u8, normalizer.type, "NFKC")) {
        normalizer.nfkc = true;
    } else if (std.mem.eql(u8, normalizer.type, "StripAccents")) {
        normalizer.strip_accents = true;
    }
    return normalizer;
}

fn parsePreTokenizer(arena_allocator: std.mem.Allocator, json_scanner: *std.json.Scanner) !schema.PreTokenizer {
    var pretokenizer = schema.PreTokenizer{};
    var split_behavior: ?[]const u8 = null;
    const first = try json_scanner.next();
    if (first == .null) return pretokenizer;
    if (first != .object_begin) return error.InvalidPreTokenizer;

    while (true) {
        const json_token = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
        switch (json_token) {
            .object_end => break,
            .allocated_string => |key| {
                if (std.mem.eql(u8, key, "type")) {
                    const type_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    if (type_value == .allocated_string) pretokenizer.type = type_value.allocated_string;
                } else if (std.mem.eql(u8, key, "behavior")) {
                    const behavior_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    if (behavior_value == .allocated_string) split_behavior = behavior_value.allocated_string;
                } else if (std.mem.eql(u8, key, "add_prefix_space")) {
                    const flag_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    pretokenizer.add_prefix_space = (flag_value == .true);
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
                    } else if (pattern_value == .object_begin) {
                        // Parse {"Regex": "..."} or {"String": "..."} format
                        while (true) {
                            const pattern_token = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                            switch (pattern_token) {
                                .object_end => break,
                                .allocated_string => |pattern_key| {
                                    if (std.mem.eql(u8, pattern_key, "Regex") or std.mem.eql(u8, pattern_key, "String")) {
                                        const pattern_field = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                                        if (pattern_field == .allocated_string) pretokenizer.pattern = pattern_field.allocated_string;
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
                    while (true) {
                        const arr_token = try json_scanner.peekNextTokenType();
                        if (arr_token == .array_end) {
                            _ = try json_scanner.next();
                            break;
                        }
                        const sub = try parsePreTokenizer(arena_allocator, json_scanner);
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
        // For Split type, behavior determines whether we emit matches or split on pattern
        // "Isolated" = emit matches (regex_split = false)
        // Other behaviors = split on pattern (regex_split = true)
        if (split_behavior) |b| {
            pretokenizer.regex_split = !std.mem.eql(u8, b, "Isolated");
        } else {
            pretokenizer.regex_split = true; // Default: split on pattern
        }
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

fn parsePostProcessor(arena_allocator: std.mem.Allocator, json_scanner: *std.json.Scanner) !schema.PostProcessor {
    var postprocessor = schema.PostProcessor{};
    const first = try json_scanner.next();
    if (first == .null) return postprocessor;
    if (first != .object_begin) return error.InvalidPostProcessor;

    while (true) {
        const json_token = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
        switch (json_token) {
            .object_end => break,
            .allocated_string => |key| {
                if (std.mem.eql(u8, key, "type")) {
                    const type_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    if (type_value == .allocated_string) postprocessor.type = type_value.allocated_string;
                } else if (std.mem.eql(u8, key, "cls")) {
                    // Parse [token, type_id] array
                    if ((try json_scanner.next()) == .array_begin) {
                        const cls_tok = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                        if (cls_tok == .allocated_string) postprocessor.cls_token = cls_tok.allocated_string;
                        try json_scanner.skipValue(); // skip type_id
                        _ = try json_scanner.next(); // array_end
                    }
                } else if (std.mem.eql(u8, key, "sep")) {
                    if ((try json_scanner.next()) == .array_begin) {
                        const sep_tok = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                        if (sep_tok == .allocated_string) postprocessor.sep_token = sep_tok.allocated_string;
                        try json_scanner.skipValue(); // skip type_id
                        _ = try json_scanner.next(); // array_end
                    }
                } else if (std.mem.eql(u8, key, "add_special_tokens")) {
                    const flag_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    postprocessor.add_special = (flag_value == .true);
                } else {
                    try json_scanner.skipValue();
                }
            },
            else => return error.InvalidPostProcessor,
        }
    }
    // Infer settings from type
    if (std.mem.eql(u8, postprocessor.type, "BertProcessing") or std.mem.eql(u8, postprocessor.type, "TemplateProcessing")) {
        postprocessor.add_special = true;
        if (postprocessor.cls_token == null) postprocessor.cls_token = "[CLS]";
        if (postprocessor.sep_token == null) postprocessor.sep_token = "[SEP]";
    } else if (std.mem.eql(u8, postprocessor.type, "RobertaProcessing")) {
        postprocessor.add_special = true;
        postprocessor.pair = true; // RoBERTa uses double SEP in pair encoding
        if (postprocessor.cls_token == null) postprocessor.cls_token = "<s>";
        if (postprocessor.sep_token == null) postprocessor.sep_token = "</s>";
    }
    return postprocessor;
}

/// Parse decoder section from tokenizer.json
/// Handles both Sequence decoder (with Strip) and simple decoders
fn parseDecoder(arena_allocator: std.mem.Allocator, json_scanner: *std.json.Scanner) !schema.Decoder {
    var decoder = schema.Decoder{};
    const first = try json_scanner.next();
    if (first == .null) return decoder;
    if (first != .object_begin) return error.InvalidDecoder;

    while (true) {
        const json_token = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
        switch (json_token) {
            .object_end => break,
            .allocated_string => |key| {
                if (std.mem.eql(u8, key, "type")) {
                    const type_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    if (type_value == .allocated_string) decoder.type = type_value.allocated_string;
                } else if (std.mem.eql(u8, key, "decoders")) {
                    // Sequence decoder - parse array of sub-decoders
                    if ((try json_scanner.next()) == .array_begin) {
                        try parseDecoderSequence(arena_allocator, json_scanner, &decoder);
                    }
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
                } else {
                    try json_scanner.skipValue();
                }
            },
            else => return error.InvalidDecoder,
        }
    }
    return decoder;
}

/// Parse decoders array within a Sequence decoder
fn parseDecoderSequence(arena_allocator: std.mem.Allocator, json_scanner: *std.json.Scanner, decoder: *schema.Decoder) !void {
    while (true) {
        const json_token = try json_scanner.next();
        switch (json_token) {
            .array_end => break,
            .object_begin => {
                // Parse sub-decoder object
                var sub_type: []const u8 = "";
                var sub_start: i32 = 0;
                var sub_stop: i32 = 0;
                while (true) {
                    const sub_token = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                    switch (sub_token) {
                        .object_end => break,
                        .allocated_string => |key| {
                            if (std.mem.eql(u8, key, "type")) {
                                const type_value = try json_scanner.nextAlloc(arena_allocator, .alloc_always);
                                if (type_value == .allocated_string) sub_type = type_value.allocated_string;
                            } else if (std.mem.eql(u8, key, "start")) {
                                const start_token = try json_scanner.next();
                                if (start_token == .number) sub_start = std.fmt.parseInt(i32, start_token.number, 10) catch 0;
                            } else if (std.mem.eql(u8, key, "stop")) {
                                const stop_token = try json_scanner.next();
                                if (stop_token == .number) sub_stop = std.fmt.parseInt(i32, stop_token.number, 10) catch 0;
                            } else {
                                try json_scanner.skipValue();
                            }
                        },
                        else => return error.InvalidDecoder,
                    }
                }
                // Apply Strip decoder settings
                if (std.mem.eql(u8, sub_type, "Strip")) {
                    decoder.strip_start = sub_start;
                    decoder.strip_stop = sub_stop;
                }
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

    // Detect model type by scanning for "type" field
    const model_type_name = detectModelType(json_bytes);

    if (std.mem.eql(u8, model_type_name, "BPE")) {
        // Use lazy BPE loader (same path as file-based loading)
        // Need to copy the JSON since lazy loader may keep references to it
        const json_copy = allocator.dupeZ(u8, json_bytes) catch return null;

        const tokenizer = ct.Tokenizer.initBpe(allocator, json_copy, true) catch {
            allocator.free(json_copy);
            return null;
        };

        // Apply added tokens and config
        applyConfigFromJson(tokenizer, json_copy) catch {
            tokenizer.destroy();
            return null;
        };
        return @ptrCast(tokenizer);
    }

    // WordPiece/Unigram: use full parsing
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const root = load_from_slice_streaming(arena.allocator(), json_bytes) catch return null;
    return build_tokenizer_from_root(&arena, root) catch null;
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

    // Detect model type by scanning for "type" field
    const model_type_name = detectModelType(json_bytes);

    if (std.mem.eql(u8, model_type_name, "BPE")) {
        // Use lazy BPE loader - defers vocab/merges parsing until first encode
        const tokenizer = ct.Tokenizer.initBpe(allocator, json_bytes, true) catch {
            allocator.free(json_bytes);
            return null;
        };
        {
            const now = std.time.nanoTimestamp();
            const duration_ms = @as(f64, @floatFromInt(now - t_start)) / 1_000_000.0;
            log.debug("tokenizer", "Lazy init", .{ .duration_ms = duration_ms }, @src());
            t_start = now;
        }

        // Apply added tokens and config (fast - these are small)
        applyConfigFromJson(tokenizer, json_bytes) catch {
            tokenizer.destroy();
            return null;
        };
        {
            const now = std.time.nanoTimestamp();
            const duration_ms = @as(f64, @floatFromInt(now - t_start)) / 1_000_000.0;
            log.debug("tokenizer", "Apply config", .{ .duration_ms = duration_ms }, @src());
        }
        return @ptrCast(tokenizer);
    }

    // WordPiece/Unigram: use full parsing
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const root = load_from_slice_streaming(arena.allocator(), json_bytes) catch {
        allocator.free(json_bytes);
        return null;
    };
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
    const result = build_tokenizer_from_root(&arena, root) catch null;
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
            try applyNormalizerFromJson(tokenizer, section[0..end], arena_allocator);
        }
    }

    // Parse pre_tokenizer
    if (findSection(json_bytes, "\"pre_tokenizer\"")) |section| {
        if (section.len > 0 and section[0] == '{') {
            const end = findMatchingBrace(section, '{', '}') orelse section.len;
            try applyPreTokenizerFromJson(tokenizer, section[0..end], arena_allocator);
        }
    }

    // Parse post_processor
    if (findSection(json_bytes, "\"post_processor\"")) |section| {
        if (section.len > 0 and section[0] == '{') {
            const end = findMatchingBrace(section, '{', '}') orelse section.len;
            applyPostProcessorFromJson(tokenizer, section[0..end]);
        }
    }

    // Parse decoder
    if (findSection(json_bytes, "\"decoder\"")) |section| {
        if (section.len > 0 and section[0] == '{') {
            const end = findMatchingBrace(section, '{', '}') orelse section.len;
            applyDecoderFromJson(tokenizer, section[0..end]);
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

        // Add token
        // Use c_allocator for persistent allocations (arena is freed on function return)
        if (token_content.len > 0) {
            const content_dup = std.heap.c_allocator.dupeZ(u8, token_content) catch return error.OutOfMemory;
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
    if (std.mem.indexOf(u8, json_bytes, field_bytes)) |pos| {
        var cursor = pos + field_bytes.len;
        // Skip whitespace and colon
        while (cursor < json_bytes.len and (json_bytes[cursor] == ' ' or json_bytes[cursor] == ':' or json_bytes[cursor] == '\t' or json_bytes[cursor] == '\n')) : (cursor += 1) {}
        if (cursor >= json_bytes.len) return null;
        const value_start = cursor;
        // Read until delimiter
        while (cursor < json_bytes.len and json_bytes[cursor] != ',' and json_bytes[cursor] != '}' and json_bytes[cursor] != ']' and json_bytes[cursor] != ' ' and json_bytes[cursor] != '\n') : (cursor += 1) {}
        return json_bytes[value_start..cursor];
    }
    return null;
}

fn findJsonFieldString(json_bytes: []const u8, field_bytes: []const u8) ?[]const u8 {
    if (std.mem.indexOf(u8, json_bytes, field_bytes)) |pos| {
        var cursor = pos + field_bytes.len;
        // Skip whitespace and colon
        while (cursor < json_bytes.len and (json_bytes[cursor] == ' ' or json_bytes[cursor] == ':' or json_bytes[cursor] == '\t' or json_bytes[cursor] == '\n')) : (cursor += 1) {}
        if (cursor >= json_bytes.len or json_bytes[cursor] != '"') return null;
        cursor += 1;
        const value_start = cursor;
        while (cursor < json_bytes.len and json_bytes[cursor] != '"') {
            if (json_bytes[cursor] == '\\') cursor += 2 else cursor += 1;
        }
        return json_bytes[value_start..cursor];
    }
    return null;
}

fn applyNormalizerFromJson(tokenizer: *ct.Tokenizer, json_bytes: []const u8, arena_allocator: std.mem.Allocator) !void {
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
                            try applyNormalizerFromJson(tokenizer, obj_content, arena_allocator);
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
    if (findJsonFieldValue(json_bytes, "\"add_prefix_space\"")) |v| {
        tokenizer.*.pretokenizer.add_prefix_space = if (std.mem.eql(u8, v, "true")) 1 else 0;
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
                        try applyPreTokenizerFromJson(tokenizer, obj, arena_allocator);
                        cursor = obj_end;
                    }
                }
            }
        } else if (std.mem.eql(u8, type_str, "ByteLevel")) {
            tokenizer.*.pretokenizer.byte_level = 1;
        } else if (std.mem.eql(u8, type_str, "Whitespace") or std.mem.eql(u8, type_str, "WhitespaceSplit")) {
            tokenizer.*.pretokenizer.whitespace = 1;
        } else if (std.mem.eql(u8, type_str, "Punctuation")) {
            tokenizer.*.pretokenizer.punctuation = 1;
        } else if (std.mem.eql(u8, type_str, "BertPreTokenizer")) {
            tokenizer.*.pretokenizer.whitespace = 1;
            tokenizer.*.pretokenizer.punctuation = 1;
        } else if (std.mem.eql(u8, type_str, "Metaspace")) {
            // Clear BPE default regex — Metaspace uses whitespace splitting, not regex
            _ = tok_fns.tokenizer_pretokenizer_set(&tokenizer.*.pretokenizer, null);
            tokenizer.*.pretokenizer.metaspace = 1;
            tokenizer.*.pretokenizer.whitespace = 1;
        } else if (std.mem.eql(u8, type_str, "Split")) {
            // Split type - parse pattern, behavior, and invert
            // Behavior: "Isolated" = emit matches, "Removed" = pattern describes what to keep (with invert)
            //           "MergedWithPrevious/Next" = split on pattern
            // Note: Don't reset byte_level here - it may be set by a sibling ByteLevel pretokenizer in a Sequence

            // Check behavior - default to MergedWithPrevious (regex_split=1)
            // For "Isolated" or "Removed", we emit the matches directly (regex_split=0)
            if (findJsonFieldString(json_bytes, "\"behavior\"")) |behavior| {
                if (std.mem.eql(u8, behavior, "Isolated") or std.mem.eql(u8, behavior, "Removed")) {
                    tokenizer.*.pretokenizer.regex_split = 0; // Emit matches
                } else {
                    tokenizer.*.pretokenizer.regex_split = 1; // Split on pattern
                }
            } else {
                tokenizer.*.pretokenizer.regex_split = 1; // Default: split on pattern
            }

            // Check invert - if true, emit matches instead of gaps
            // With invert=true, the regex pattern describes what tokens to KEEP
            if (findJsonFieldString(json_bytes, "\"invert\"")) |invert_val| {
                if (std.mem.eql(u8, invert_val, "true")) {
                    tokenizer.*.pretokenizer.regex_invert = 1;
                    tokenizer.*.pretokenizer.regex_split = 0; // With invert, we emit matches
                }
            }

            // Parse pattern - can be {"String": " "} or {"Regex": "..."}
            // Use c_allocator for persistent allocations (arena is freed on function return)
            if (findSection(json_bytes, "\"pattern\"")) |pattern_section| {
                if (pattern_section.len > 0 and pattern_section[0] == '{') {
                    // Object format: {"String": " "} or {"Regex": "..."}
                    if (findJsonFieldString(pattern_section, "\"String\"")) |str_pattern| {
                        // Unescape JSON string and compile pattern as regex
                        const unescaped = try json_utils.unescapeJsonString(arena_allocator, str_pattern);
                        const pat_z = std.heap.c_allocator.dupeZ(u8, unescaped) catch return error.OutOfMemory;
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
fn applyPostProcessorFromJson(tokenizer: *ct.Tokenizer, json_bytes: []const u8) void {
    const type_str = findJsonFieldString(json_bytes, "\"type\"") orelse return;

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
        cls_str = findJsonFieldString(json_bytes, "\"cls\"");
        sep_str = findJsonFieldString(json_bytes, "\"sep\"");
    }

    // For TemplateProcessing: extract from special_tokens section
    if (std.mem.eql(u8, type_str, "TemplateProcessing")) {
        if (findSection(json_bytes, "\"special_tokens\"")) |st_section| {
            // Find token strings by looking for "id" fields
            // The special_tokens map has entries like: "<s>": {"id": "<s>", "ids": [1], ...}
            // We extract the keys (which are the token strings)
            var cursor: usize = 0;
            var found_first = false;
            while (cursor < st_section.len) {
                if (st_section[cursor] == '"') {
                    const str_start = cursor + 1;
                    cursor += 1;
                    while (cursor < st_section.len and st_section[cursor] != '"') {
                        if (st_section[cursor] == '\\') cursor += 2 else cursor += 1;
                    }
                    const str_end = cursor;
                    if (cursor < st_section.len) cursor += 1;
                    // Check if this is followed by ':' (a key, not a value)
                    var peek = cursor;
                    while (peek < st_section.len and (st_section[peek] == ' ' or st_section[peek] == '\n' or st_section[peek] == '\t')) : (peek += 1) {}
                    if (peek < st_section.len and st_section[peek] == ':') {
                        const key = st_section[str_start..str_end];
                        if (!found_first) {
                            cls_str = key;
                            found_first = true;
                        } else if (sep_str == null) {
                            sep_str = key;
                        }
                        cursor = peek + 1;
                        // Skip past the value object to avoid picking up nested keys
                        while (cursor < st_section.len and st_section[cursor] != '{') : (cursor += 1) {}
                        if (cursor < st_section.len) {
                            cursor = cursor + (findMatchingBrace(st_section[cursor..], '{', '}') orelse (st_section.len - cursor));
                        }
                    }
                } else {
                    cursor += 1;
                }
            }
        }
    }

    // Default token strings
    if (cls_str == null) {
        if (std.mem.eql(u8, type_str, "RobertaProcessing")) {
            cls_str = "<s>";
            sep_str = if (sep_str == null) "</s>" else sep_str;
        } else {
            cls_str = "[CLS]";
            sep_str = if (sep_str == null) "[SEP]" else sep_str;
        }
    }
    if (sep_str == null) sep_str = cls_str;

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

fn applyDecoderFromJson(tokenizer: *ct.Tokenizer, json_bytes: []const u8) void {
    // Look for Strip decoder by finding "type": "Strip" pattern
    const strip_pattern = "\"type\": \"Strip\"";
    const strip_pattern2 = "\"type\":\"Strip\"";

    // Find Strip decoder in the JSON
    const strip_pos = std.mem.indexOf(u8, json_bytes, strip_pattern) orelse
        std.mem.indexOf(u8, json_bytes, strip_pattern2) orelse return;

    // Find the decoder object boundaries (look backward for '{')
    var obj_start = strip_pos;
    while (obj_start > 0 and json_bytes[obj_start] != '{') : (obj_start -= 1) {}

    // Find closing brace
    const obj_slice = json_bytes[obj_start..];
    const obj_end = findMatchingBrace(obj_slice, '{', '}') orelse return;
    const strip_obj = obj_slice[0..obj_end];

    // Extract "start" value from the Strip object
    if (findJsonFieldValue(strip_obj, "\"start\"")) |start_str| {
        tokenizer.decoder.strip_start = std.fmt.parseInt(i32, start_str, 10) catch 0;
    }
}

fn build_tokenizer_from_root(arena: *std.heap.ArenaAllocator, root: schema.TokenizerRoot) !*ct.Tokenizer {
    const model_type_name = root.model.type;
    var tokenizer_opt: ?*ct.Tokenizer = null;
    if (std.mem.eql(u8, model_type_name, "BPE")) {
        tokenizer_opt = try build_bpe(arena, root.model);
    } else if (std.mem.eql(u8, model_type_name, "WordPiece")) {
        tokenizer_opt = try build_wordpiece(arena, root.model);
    } else if (std.mem.eql(u8, model_type_name, "Unigram")) {
        tokenizer_opt = try build_unigram(arena, root.model);
    } else {
        return error.UnsupportedModel;
    }
    const tokenizer = tokenizer_opt orelse return error.BuildFailed;
    try apply_added_tokens(tokenizer, root.added_tokens);

    // Apply normalizer settings
    const normalizer_spec = ct.NormalizerSpec{
        .type = if (root.normalizer.type.len > 0) (arena.allocator().dupeZ(u8, root.normalizer.type) catch return error.BuildFailed).ptr else null,
        .lowercase = if (root.normalizer.lowercase) 1 else 0,
        .strip_accents = if (root.normalizer.strip_accents) 1 else 0,
        .nfc = if (root.normalizer.nfc) 1 else 0,
        .nfd = if (root.normalizer.nfd) 1 else 0,
        .nfkc = if (root.normalizer.nfkc) 1 else 0,
        .clean_text = if (root.normalizer.clean_text) 1 else 0,
        .handle_chinese_chars = if (root.normalizer.handle_chinese_chars) 1 else 0,
        // SentencePiece-style normalizers
        .prepend = if (root.normalizer.prepend) |p| (arena.allocator().dupeZ(u8, p) catch return error.BuildFailed).ptr else null,
        .replace_pattern = if (root.normalizer.replace_pattern) |p| (arena.allocator().dupeZ(u8, p) catch return error.BuildFailed).ptr else null,
        .replace_content = if (root.normalizer.replace_content) |c_val| (arena.allocator().dupeZ(u8, c_val) catch return error.BuildFailed).ptr else null,
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
    }

    // Resolve cls_id/sep_id from added tokens when post_processor is active.
    // BPE/Unigram init sets cls_id/sep_id to -1; WordPiece resolves from its own
    // vocab. For TemplateProcessing/BertProcessing on BPE models, the token
    // strings (cls_token/sep_token) are set above but the IDs are never resolved.
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

    // Apply decoder settings (e.g., Strip decoder for SentencePiece)
    tokenizer.decoder.strip_start = @intCast(root.decoder.strip_start);
    tokenizer.decoder.strip_stop = @intCast(root.decoder.strip_stop);
    tokenizer.decoder.add_prefix_space = if (root.decoder.add_prefix_space) 1 else 0;

    return tokenizer;
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
    const result = @as(*ct.Tokenizer, @ptrCast(bpe.tokenizer_bpe_create_from_spec(@ptrCast(&model_spec))));
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
    };
    return @ptrCast(wordpiece_model.tokenizer_wordpiece_create_from_spec(@ptrCast(&model_spec)));
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
    return @ptrCast(unigram_model.tokenizer_unigram_create_from_spec(@ptrCast(&model_spec)));
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

test "findJsonFieldValue extracts number" {
    const json = \\{"id": 123, "name": "test"}
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
    const json = \\{"name": "test", "count": 42}
    ;
    const value = findJsonFieldValue(json, "\"name\"");
    try std.testing.expect(value != null);
    try std.testing.expectEqualStrings("\"test\"", value.?);
}

test "findJsonFieldValue extracts boolean true" {
    const json = \\{"enabled": true, "other": false}
    ;
    const value = findJsonFieldValue(json, "\"enabled\"");
    try std.testing.expect(value != null);
    try std.testing.expectEqualStrings("true", value.?);
}

test "findJsonFieldValue extracts boolean false" {
    const json = \\{"enabled": false}
    ;
    const value = findJsonFieldValue(json, "\"enabled\"");
    try std.testing.expect(value != null);
    try std.testing.expectEqualStrings("false", value.?);
}

test "findJsonFieldValue returns null for missing field" {
    const json = \\{"name": "test"}
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
    const json = \\{"type": "BPE", "name": "tokenizer"}
    ;
    const value = findJsonFieldString(json, "\"type\"");
    try std.testing.expect(value != null);
    try std.testing.expectEqualStrings("BPE", value.?);
}

test "findJsonFieldString extracts value without outer quotes" {
    const json = \\{"path": "test_value"}
    ;
    const value = findJsonFieldString(json, "\"path\"");
    try std.testing.expect(value != null);
    try std.testing.expectEqualStrings("test_value", value.?);
}

test "findJsonFieldString returns null for non-string value" {
    const json = \\{"count": 42}
    ;
    const value = findJsonFieldString(json, "\"count\"");
    try std.testing.expect(value == null);
}

test "findJsonFieldString returns null for missing field" {
    const json = \\{"name": "test"}
    ;
    const value = findJsonFieldString(json, "\"missing\"");
    try std.testing.expect(value == null);
}

test "findQuotedString finds first quoted string" {
    const input = \\some text "hello world" more text
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

test "tokenizer_loader_from_json_string requires integration testing" {
    // This function requires:
    // - Complete JSON tokenizer specification
    // - Model type detection and routing
    // - Lazy/full parsing based on model type
    // - Configuration application
    // Integration tests: tests/tokenizer/test_*.py
}
