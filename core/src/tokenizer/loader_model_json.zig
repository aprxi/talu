//! Tokenizer loader model-section JSON helpers.
//!
//! Owns tokenizer.json model option validation plus fast vocab/merge parsing.

const std = @import("std");
const schema = @import("schema.zig");
const utils = @import("utils.zig");
const json_utils = @import("json_utils.zig");
const log = @import("log_pkg");

const ManagedArrayList = std.array_list.Managed;
const findMatchingBrace = utils.findMatchingBrace;
const findJsonFieldValue = json_utils.findJsonFieldValue;

pub fn validateModelOptions(model_type_name: []const u8, model_json: []const u8) !void {
    if (!std.mem.eql(u8, model_type_name, "BPE")) return;

    // Accept HuggingFace's inert defaults, but reject options that would alter
    // runtime behavior without an explicit implementation.
    if (findJsonFieldValue(model_json, "\"dropout\"")) |value| {
        if (!std.mem.eql(u8, value, "null")) {
            log.debug("tokenizer", "Tokenizer BPE option rejected", .{
                .field = "dropout",
                .value_len = value.len,
                .value_preview = value[0..@min(value.len, 32)],
            }, @src());
            return error.InvalidModel;
        }
    }

    var fuse_unk_enabled = false;
    if (findJsonFieldValue(model_json, "\"fuse_unk\"")) |value| {
        if (std.mem.eql(u8, value, "true")) {
            fuse_unk_enabled = true;
        } else if (!std.mem.eql(u8, value, "false")) {
            log.debug("tokenizer", "Tokenizer BPE option rejected", .{
                .field = "fuse_unk",
                .value_len = value.len,
                .value_preview = value[0..@min(value.len, 32)],
            }, @src());
            return error.InvalidModel;
        }
    }

    var byte_fallback_enabled = false;
    if (findJsonFieldValue(model_json, "\"byte_fallback\"")) |value| {
        if (std.mem.eql(u8, value, "true")) {
            byte_fallback_enabled = true;
        } else if (!std.mem.eql(u8, value, "false")) {
            log.debug("tokenizer", "Tokenizer BPE option rejected", .{
                .field = "byte_fallback",
                .value_len = value.len,
                .value_preview = value[0..@min(value.len, 32)],
            }, @src());
            return error.InvalidModel;
        }
    }

    if (fuse_unk_enabled and !byte_fallback_enabled) {
        log.debug("tokenizer", "Tokenizer BPE option rejected", .{
            .field = "fuse_unk",
            .reason = "requires_byte_fallback",
        }, @src());
        return error.InvalidModel;
    }

    if (findJsonFieldValue(model_json, "\"continuing_subword_prefix\"")) |value| {
        if (!std.mem.eql(u8, value, "null") and !std.mem.eql(u8, value, "\"\"")) {
            log.debug("tokenizer", "Tokenizer BPE option rejected", .{
                .field = "continuing_subword_prefix",
                .value_len = value.len,
                .value_preview = value[0..@min(value.len, 32)],
            }, @src());
            return error.InvalidModel;
        }
    }

    if (findJsonFieldValue(model_json, "\"end_of_word_suffix\"")) |value| {
        if (!std.mem.eql(u8, value, "null") and !std.mem.eql(u8, value, "\"\"")) {
            log.debug("tokenizer", "Tokenizer BPE option rejected", .{
                .field = "end_of_word_suffix",
                .value_len = value.len,
                .value_preview = value[0..@min(value.len, 32)],
            }, @src());
            return error.InvalidModel;
        }
    }

    if (findJsonFieldValue(model_json, "\"ignore_merges\"")) |value| {
        if (!std.mem.eql(u8, value, "false") and !std.mem.eql(u8, value, "true")) {
            log.debug("tokenizer", "Tokenizer BPE option rejected", .{
                .field = "ignore_merges",
                .value_len = value.len,
                .value_preview = value[0..@min(value.len, 32)],
            }, @src());
            return error.InvalidModel;
        }
    }
}

pub fn validateBpeMergeReferences(
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

        // Later merge rules may reference outputs created by earlier rules.
        const merged = try std.mem.concat(arena_allocator, u8, &.{ lhs, rhs });
        try vocab_tokens.put(merged, {});
    }
}

pub fn findQuotedString(json_bytes: []const u8) ?[]const u8 {
    var cursor: usize = 0;
    while (cursor < json_bytes.len and json_bytes[cursor] != '"') : (cursor += 1) {}
    if (cursor >= json_bytes.len) return null;
    cursor += 1;
    const start = cursor;
    while (cursor < json_bytes.len and json_bytes[cursor] != '"') {
        if (json_bytes[cursor] == '\\') cursor += 2 else cursor += 1;
    }
    if (cursor >= json_bytes.len) return null;
    return json_bytes[start..cursor];
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
                if (json_bytes[cursor] == '\\') cursor += 2 else cursor += 1;
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

pub fn parseMergesFastSection(
    arena_allocator: std.mem.Allocator,
    json_bytes: []const u8,
    merge_entries: *ManagedArrayList([]const u8),
) !void {
    try merge_entries.ensureTotalCapacity(600000);
    var seen_merges = std.StringHashMap(void).init(arena_allocator);
    defer seen_merges.deinit();

    var cursor: usize = 0;
    while (cursor < json_bytes.len and std.ascii.isWhitespace(json_bytes[cursor])) : (cursor += 1) {}
    if (cursor < json_bytes.len and json_bytes[cursor] == '[') cursor += 1;
    while (cursor < json_bytes.len) {
        while (cursor < json_bytes.len and json_bytes[cursor] != '[' and json_bytes[cursor] != '"') : (cursor += 1) {}
        if (cursor >= json_bytes.len) break;

        if (json_bytes[cursor] == '[') {
            cursor += 1;
            while (cursor < json_bytes.len and std.ascii.isWhitespace(json_bytes[cursor])) : (cursor += 1) {}
            if (cursor < json_bytes.len and json_bytes[cursor] == ']') {
                cursor += 1;
                continue;
            }

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
            if (cursor >= json_bytes.len) return error.InvalidMerges;
            const a_end = cursor;
            cursor += 1;
            while (cursor < json_bytes.len and (std.ascii.isWhitespace(json_bytes[cursor]) or json_bytes[cursor] == ',')) : (cursor += 1) {}

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
            if (cursor >= json_bytes.len) return error.InvalidMerges;
            const b_end = cursor;
            cursor += 1;

            const lhs = if (a_escape) try unescapeJsonStringFast(arena_allocator, json_bytes[a_start..a_end]) else json_bytes[a_start..a_end];
            const rhs = if (b_escape) try unescapeJsonStringFast(arena_allocator, json_bytes[b_start..b_end]) else json_bytes[b_start..b_end];
            if (lhs.len == 0 or rhs.len == 0 or std.mem.indexOfScalar(u8, lhs, 0) != null or std.mem.indexOfScalar(u8, rhs, 0) != null) {
                return error.InvalidMerges;
            }

            const merged = try std.fmt.allocPrint(arena_allocator, "{s} {s}", .{ lhs, rhs });
            if (seen_merges.contains(merged)) return error.InvalidMerges;
            try seen_merges.put(merged, {});
            merge_entries.appendAssumeCapacity(merged);

            while (cursor < json_bytes.len and std.ascii.isWhitespace(json_bytes[cursor])) : (cursor += 1) {}
            if (cursor >= json_bytes.len or json_bytes[cursor] != ']') return error.InvalidMerges;
            cursor += 1;
        } else {
            cursor += 1;
            const start = cursor;
            var has_escape = false;
            while (cursor < json_bytes.len and json_bytes[cursor] != '"') {
                if (json_bytes[cursor] == '\\') {
                    has_escape = true;
                    cursor += 2;
                } else cursor += 1;
            }
            if (cursor >= json_bytes.len) return error.InvalidMerges;

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

pub fn parseVocabFastSection(
    arena_allocator: std.mem.Allocator,
    json_bytes: []const u8,
    vocab_entries: *ManagedArrayList(schema.TokenId),
) !void {
    try vocab_entries.ensureTotalCapacity(300000);
    var seen_tokens = std.StringHashMap(void).init(arena_allocator);
    defer seen_tokens.deinit();
    var seen_ids = std.AutoHashMap(i32, void).init(arena_allocator);
    defer seen_ids.deinit();

    var cursor: usize = 0;
    while (cursor < json_bytes.len) {
        while (cursor < json_bytes.len and json_bytes[cursor] != '"') : (cursor += 1) {}
        if (cursor >= json_bytes.len) break;
        cursor += 1;

        const key_start = cursor;
        var has_escape = false;
        while (cursor < json_bytes.len and json_bytes[cursor] != '"') {
            if (json_bytes[cursor] == '\\') {
                has_escape = true;
                cursor += 2;
            } else {
                cursor += 1;
            }
        }
        if (cursor >= json_bytes.len) return error.InvalidVocab;
        const key_end = cursor;
        cursor += 1;

        while (cursor < json_bytes.len and (json_bytes[cursor] == ' ' or json_bytes[cursor] == ':' or json_bytes[cursor] == '\t' or json_bytes[cursor] == '\n' or json_bytes[cursor] == '\r')) : (cursor += 1) {}
        if (cursor >= json_bytes.len) break;

        const ch = json_bytes[cursor];
        if (ch == '-') return error.InvalidVocab;
        if (ch >= '0' and ch <= '9') {
            const num_start = cursor;
            while (cursor < json_bytes.len and json_bytes[cursor] >= '0' and json_bytes[cursor] <= '9') : (cursor += 1) {}
            if (cursor < json_bytes.len and (json_bytes[cursor] == '.' or json_bytes[cursor] == 'e' or json_bytes[cursor] == 'E')) {
                return error.InvalidVocab;
            }
            const id_num = std.fmt.parseInt(i32, json_bytes[num_start..cursor], 10) catch return error.InvalidVocab;

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

pub fn unescapeJsonStringFast(arena_allocator: std.mem.Allocator, input_bytes: []const u8) ![]const u8 {
    const unescaped = try json_utils.unescapeJsonString(arena_allocator, input_bytes);
    if (unescaped.ptr == input_bytes.ptr and unescaped.len == input_bytes.len) {
        return try arena_allocator.dupe(u8, input_bytes);
    }
    return unescaped;
}
