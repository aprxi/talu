//! Tokenizer loader parser helpers.
//!
//! Parses tokenizer.json metadata sections that configure added tokens and
//! pipeline behavior.

const std = @import("std");
const schema = @import("schema.zig");
const ct = @import("c_types.zig");
const tok_fns = @import("pipeline.zig");
const utils = @import("utils.zig");
const json_utils = @import("json_utils.zig");
const log = @import("log_pkg");

const ManagedArrayList = std.array_list.Managed;
const findSection = utils.findJsonSection;
const findMatchingBrace = utils.findMatchingBrace;
const findJsonFieldString = json_utils.findJsonFieldString;

pub fn parseMetadataSections(
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

pub const MAX_JSON_PIPELINE_DEPTH: usize = 128;
const MAX_JSON_PIPELINE_BREADTH: usize = 16_384;

pub fn ensureJsonDepthWithinLimit(json_bytes: []const u8, comptime err: anyerror) !void {
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

pub fn parseNormalizer(arena_allocator: std.mem.Allocator, json_scanner: *std.json.Scanner) !schema.Normalizer {
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
        log.debug("tokenizer", "Normalizer parse rejected non-object token", .{
            .token = @tagName(first_token),
        }, @src());
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
                    var element_count: usize = 0;
                    while (true) {
                        const arr_token = try json_scanner.peekNextTokenType();
                        if (arr_token == .array_end) {
                            _ = try json_scanner.next();
                            break;
                        }
                        element_count += 1;
                        if (element_count > MAX_JSON_PIPELINE_BREADTH) return error.InvalidNormalizer;
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

pub fn parsePreTokenizer(arena_allocator: std.mem.Allocator, json_scanner: *std.json.Scanner) !schema.PreTokenizer {
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
    var saw_regex_pattern = false;
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
                                        if (std.mem.eql(u8, pattern_key, "Regex")) saw_regex_pattern = true;
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
                    var element_count: usize = 0;
                    while (true) {
                        const arr_token = try json_scanner.peekNextTokenType();
                        if (arr_token == .array_end) {
                            _ = try json_scanner.next();
                            break;
                        }
                        element_count += 1;
                        if (element_count > MAX_JSON_PIPELINE_BREADTH) return error.InvalidPreTokenizer;
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
        if (saw_regex_pattern) {
            var regex_validator = std.mem.zeroes(ct.PreTokenizer);
            defer tok_fns.tokenizer_pretokenizer_free(&regex_validator);
            const pat_z = try arena_allocator.dupeZ(u8, pretokenizer.pattern.?);
            if (tok_fns.tokenizer_pretokenizer_set(&regex_validator, pat_z.ptr) != 0) {
                return error.InvalidPreTokenizer;
            }
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

pub fn parsePostProcessor(arena_allocator: std.mem.Allocator, json_scanner: *std.json.Scanner) !schema.PostProcessor {
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

pub fn parseDecoder(arena_allocator: std.mem.Allocator, json_scanner: *std.json.Scanner) !schema.Decoder {
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
