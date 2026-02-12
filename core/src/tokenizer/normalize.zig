//! Text Normalization
//!
//! Unicode normalization (NFC/NFD/NFKC) and text preprocessing.
//! Uses utf8proc for Unicode operations.

const std = @import("std");
const ct = @import("c_types.zig");
const types = @import("types.zig");

const c = @cImport({
    @cInclude("utf8proc.h");
});

const Allocator = types.Allocator;
const Normalized = types.Normalized;

pub const NormalizeError = error{
    OutOfMemory,
    Utf8ProcFailed,
};

fn stripAccents(codepoint: c.utf8proc_int32_t) c.utf8proc_int32_t {
    var decomposed: [4]c.utf8proc_int32_t = undefined;
    var last_boundary_class: c_int = 0;
    const options: c.utf8proc_option_t = c.UTF8PROC_STRIPMARK | c.UTF8PROC_COMPOSE;
    const written_count = c.utf8proc_decompose_char(codepoint, &decomposed, decomposed.len, options, &last_boundary_class);
    if (written_count <= 0) return codepoint;
    return decomposed[0];
}

pub fn tokenizer_apply_normalizer_spec(tok: ?*ct.Tokenizer, spec: ?*const ct.NormalizerSpec) void {
    if (tok == null or spec == null) return;
    const tokenizer = tok.?;
    const normalizer_spec = spec.?;
    tokenizer.normalizer.lowercase = normalizer_spec.lowercase;
    tokenizer.normalizer.strip_accents = normalizer_spec.strip_accents;
    tokenizer.normalizer.nfc = normalizer_spec.nfc;
    tokenizer.normalizer.nfd = normalizer_spec.nfd;
    tokenizer.normalizer.nfkc = normalizer_spec.nfkc;
    tokenizer.normalizer.clean_text = normalizer_spec.clean_text;
    tokenizer.normalizer.handle_chinese_chars = normalizer_spec.handle_chinese_chars;
    // SentencePiece-style normalizers
    tokenizer.normalizer.prepend = normalizer_spec.prepend;
    tokenizer.normalizer.replace_pattern = normalizer_spec.replace_pattern;
    tokenizer.normalizer.replace_content = normalizer_spec.replace_content;
}

/// Apply NFC normalization while preserving embedded null bytes.
/// utf8proc_NFC stops at null bytes, so we split input at nulls,
/// NFC each segment, and reassemble with nulls preserved.
fn applyNfcWithNullBytes(input_bytes: []const u8) NormalizeError![]u8 {
    // Fast path: no null bytes in input
    if (std.mem.indexOfScalar(u8, input_bytes, 0) == null) {
        // No embedded nulls - use simple NFC
        const input_copy = try Allocator.alloc(u8, input_bytes.len + 1);
        errdefer Allocator.free(input_copy);
        @memcpy(input_copy[0..input_bytes.len], input_bytes);
        input_copy[input_bytes.len] = 0;
        const nfc_result_ptr = c.utf8proc_NFC(@ptrCast(input_copy.ptr));
        Allocator.free(input_copy);
        if (nfc_result_ptr == null) return error.Utf8ProcFailed;
        const nfc_result = nfc_result_ptr.?;
        defer std.c.free(nfc_result);
        var nfc_length: usize = 0;
        while (nfc_result[nfc_length] != 0) : (nfc_length += 1) {}
        const output_bytes = try Allocator.alloc(u8, nfc_length);
        @memcpy(output_bytes, nfc_result[0..nfc_length]);
        return output_bytes;
    }

    // Slow path: handle embedded null bytes
    var output_bytes = std.ArrayListUnmanaged(u8){};
    errdefer output_bytes.deinit(Allocator);

    var cursor: usize = 0;
    while (cursor < input_bytes.len) {
        // Find next null byte or end
        var segment_end = cursor;
        while (segment_end < input_bytes.len and input_bytes[segment_end] != 0) : (segment_end += 1) {}

        if (segment_end > cursor) {
            // NFC the segment (cursor..segment_end)
            const segment_bytes = input_bytes[cursor..segment_end];
            const segment_copy = try Allocator.alloc(u8, segment_bytes.len + 1);
            defer Allocator.free(segment_copy);
            @memcpy(segment_copy[0..segment_bytes.len], segment_bytes);
            segment_copy[segment_bytes.len] = 0;

            const nfc_ptr = c.utf8proc_NFC(@ptrCast(segment_copy.ptr));
            if (nfc_ptr == null) return error.Utf8ProcFailed;
            const nfc_segment = nfc_ptr.?;
            defer std.c.free(nfc_segment);
            var nfc_len: usize = 0;
            while (nfc_segment[nfc_len] != 0) : (nfc_len += 1) {}
            try output_bytes.appendSlice(Allocator, nfc_segment[0..nfc_len]);
        }

        // Append null byte if we're at one
        if (segment_end < input_bytes.len and input_bytes[segment_end] == 0) {
            try output_bytes.append(Allocator, 0);
            segment_end += 1;
        }
        cursor = segment_end;
    }

    return try output_bytes.toOwnedSlice(Allocator);
}

pub fn normalize_text(normalizer: *const ct.Normalizer, input_bytes: []const u8) NormalizeError!Normalized {
    const sentencepiece = try applySentencePieceTransforms(normalizer, input_bytes);
    defer if (sentencepiece.owned_text) |owned_text| Allocator.free(owned_text);
    defer if (sentencepiece.map) |map_values| Allocator.free(map_values);

    var normalized_bytes = sentencepiece.text;

    // Apply NFC normalization if enabled (compose combining characters)
    // Note: utf8proc_NFC expects null-terminated input and stops at embedded nulls.
    // We handle this by splitting at null bytes, NFC'ing each segment, and reassembling.
    var nfc_bytes: ?[]u8 = null;
    if (normalizer.nfc != 0) {
        nfc_bytes = try applyNfcWithNullBytes(normalized_bytes);
        normalized_bytes = nfc_bytes.?;
    }
    defer if (nfc_bytes) |owned_nfc| Allocator.free(owned_nfc);

    const normalized_capacity = normalized_bytes.len * 4 + 8;
    var normalized_buf = try Allocator.alloc(u8, normalized_capacity);
    var position_map = Allocator.alloc(i32, normalized_capacity) catch {
        Allocator.free(normalized_buf);
        return error.OutOfMemory;
    };

    var out_index: usize = 0;
    var input_index: usize = 0;

    while (input_index < normalized_bytes.len) {
        var codepoint: c.utf8proc_int32_t = 0;
        const consumed_len = c.utf8proc_iterate(@ptrCast(normalized_bytes.ptr + input_index), @intCast(normalized_bytes.len - input_index), &codepoint);
        if (consumed_len <= 0) {
            input_index += 1;
            continue;
        }

        // Map position in normalized_bytes to position in original input
        // If sp_map exists, use it to get the true original position
        const original_pos: i32 = if (sentencepiece.map) |m| m[input_index] else @intCast(input_index);

        input_index += @intCast(consumed_len);

        // Clean text: drop control, normalize whitespace to space
        if (normalizer.clean_text != 0) {
            if (codepoint == 0 or codepoint == 0xFF or codepoint < 32) continue;
            if (c.utf8proc_category(codepoint) == c.UTF8PROC_CATEGORY_ZS or std.ascii.isWhitespace(@intCast(@as(u32, @bitCast(codepoint)) & 0xFF))) {
                codepoint = ' ';
            }
        }

        const is_cjk = (codepoint >= 0x4E00 and codepoint <= 0x9FFF);

        // Lowercase
        if (normalizer.lowercase != 0) {
            codepoint = c.utf8proc_tolower(codepoint);
        }

        // Strip accents
        if (normalizer.strip_accents != 0) {
            codepoint = stripAccents(codepoint);
        }

        var utf8_bytes: [4]u8 = undefined;
        const utf8_len = std.unicode.utf8Encode(@intCast(codepoint), &utf8_bytes) catch continue;

        // Apply handle_chinese_chars: add spaces around CJK
        if (normalizer.handle_chinese_chars != 0 and is_cjk) {
            normalized_buf[out_index] = ' ';
            position_map[out_index] = original_pos;
            out_index += 1;
        }

        @memcpy(normalized_buf[out_index..][0..utf8_len], utf8_bytes[0..utf8_len]);
        for (0..utf8_len) |byte_idx| position_map[out_index + byte_idx] = original_pos;
        out_index += utf8_len;

        if (normalizer.handle_chinese_chars != 0 and is_cjk) {
            normalized_buf[out_index] = ' ';
            position_map[out_index] = original_pos;
            out_index += 1;
        }
    }

    // Strip left/right whitespace
    var trim_start: usize = 0;
    var trim_end: usize = out_index;
    if (normalizer.strip_left != 0) {
        while (trim_start < trim_end and std.ascii.isWhitespace(normalized_buf[trim_start])) trim_start += 1;
    }
    if (normalizer.strip_right != 0) {
        while (trim_end > trim_start and std.ascii.isWhitespace(normalized_buf[trim_end - 1])) trim_end -= 1;
    }
    const trimmed_len = trim_end - trim_start;
    if (trim_start > 0 and trimmed_len > 0) {
        std.mem.copyForwards(u8, normalized_buf[0..trimmed_len], normalized_buf[trim_start..trim_end]);
        std.mem.copyForwards(i32, position_map[0..trimmed_len], position_map[trim_start..trim_end]);
    }

    return Normalized{
        .text = normalized_buf[0..trimmed_len],
        .map = position_map[0..trimmed_len],
    };
}

const SentencePieceTransform = struct {
    text: []const u8,
    owned_text: ?[]u8,
    map: ?[]i32,
};

fn applySentencePieceTransforms(normalizer: *const ct.Normalizer, input_bytes: []const u8) NormalizeError!SentencePieceTransform {
    // Apply SentencePiece-style prepend and replace normalizers first
    // We also build a position map from normalized positions to original positions
    const has_prepend = normalizer.prepend != null;
    const has_replace = normalizer.replace_pattern != null and normalizer.replace_content != null;
    if (!has_prepend and !has_replace) {
        return .{ .text = input_bytes, .owned_text = null, .map = null };
    }

    // Calculate required size
    const prepend_bytes = if (normalizer.prepend) |p| std.mem.sliceTo(p, 0) else "";
    const replace_pattern = if (normalizer.replace_pattern) |p| std.mem.sliceTo(p, 0) else "";
    const replace_content = if (normalizer.replace_content) |c_ptr| std.mem.sliceTo(c_ptr, 0) else "";

    // Estimate max size: prepend + input with all patterns replaced
    const max_replace_count = if (replace_pattern.len > 0) input_bytes.len / replace_pattern.len + 1 else 0;
    const replace_size_delta = if (replace_content.len > replace_pattern.len) replace_content.len - replace_pattern.len else 0;
    const estimated_size = prepend_bytes.len + input_bytes.len + max_replace_count * replace_size_delta + 16;

    const sp_buffer = try Allocator.alloc(u8, estimated_size);
    errdefer Allocator.free(sp_buffer);
    const sp_position_map = try Allocator.alloc(i32, estimated_size);
    errdefer Allocator.free(sp_position_map);
    var sp_out = sp_buffer;
    var sp_pos_map = sp_position_map;
    var sp_index: usize = 0;

    // Apply prepend - these positions map to -1 (no original position)
    if (has_prepend) {
        for (0..prepend_bytes.len) |_| {
            sp_out[sp_index] = prepend_bytes[sp_index];
            sp_pos_map[sp_index] = -1; // No original position for prepended chars
            sp_index += 1;
        }
    }

    // Apply replace (simple string replacement, not regex)
    // Track original position for each output position
    if (has_replace and replace_pattern.len > 0) {
        var original_index: usize = 0;
        while (original_index < input_bytes.len) {
            if (original_index + replace_pattern.len <= input_bytes.len and std.mem.eql(u8, input_bytes[original_index..][0..replace_pattern.len], replace_pattern)) {
                // Replacement - map first char of replacement to original position
                for (0..replace_content.len) |replace_index| {
                    sp_out[sp_index] = replace_content[replace_index];
                    sp_pos_map[sp_index] = if (replace_index == 0) @intCast(original_index) else -1;
                    sp_index += 1;
                }
                original_index += replace_pattern.len;
            } else {
                sp_out[sp_index] = input_bytes[original_index];
                sp_pos_map[sp_index] = @intCast(original_index);
                sp_index += 1;
                original_index += 1;
            }
        }
    } else {
        // Just copy input after prepend - map each position to original
        for (0..input_bytes.len) |original_index| {
            sp_out[sp_index] = input_bytes[original_index];
            sp_pos_map[sp_index] = @intCast(original_index);
            sp_index += 1;
        }
    }

    return .{
        .text = sp_out[0..sp_index],
        .owned_text = sp_buffer,
        .map = sp_position_map,
    };
}

pub fn addPrefixSpace(normalized: *Normalized) !void {
    const new_len = normalized.text.len + 1;
    const prefixed_bytes = try Allocator.alloc(u8, new_len);
    errdefer Allocator.free(prefixed_bytes);
    const prefixed_map = try Allocator.alloc(i32, new_len);

    prefixed_bytes[0] = ' ';
    prefixed_map[0] = -1;
    if (normalized.text.len > 0) {
        @memcpy(prefixed_bytes[1..], normalized.text);
        @memcpy(prefixed_map[1..], normalized.map);
    }
    Allocator.free(normalized.text);
    Allocator.free(normalized.map);
    normalized.text = prefixed_bytes;
    normalized.map = prefixed_map;
}

// =============================================================================
// Tests
// =============================================================================

// Note: normalize_text requires full tokenizer context and C library
// dependencies (utf8proc). It is tested via integration tests in
// tests/tokenizer/. Helper functions like addPrefixSpace and
// applyNfcWithNullBytes are unit-testable.

test "addPrefixSpace adds space at beginning" {
    var normalized = Normalized{
        .text = try Allocator.dupe(u8, "hello"),
        .map = try Allocator.dupe(i32, &[_]i32{ 0, 1, 2, 3, 4 }),
    };

    try addPrefixSpace(&normalized);
    defer normalized.deinit();

    try std.testing.expectEqual(@as(usize, 6), normalized.text.len);
    try std.testing.expectEqualStrings(" hello", normalized.text);
    try std.testing.expectEqual(@as(i32, -1), normalized.map[0]);
    try std.testing.expectEqual(@as(i32, 0), normalized.map[1]);
}

test "addPrefixSpace handles empty string" {
    var normalized = Normalized{
        .text = try Allocator.alloc(u8, 0),
        .map = try Allocator.alloc(i32, 0),
    };

    try addPrefixSpace(&normalized);
    defer normalized.deinit();

    try std.testing.expectEqual(@as(usize, 1), normalized.text.len);
    try std.testing.expectEqual(@as(u8, ' '), normalized.text[0]);
}

test "applyNfcWithNullBytes handles text without nulls" {
    const input = "hello";
    const result = try applyNfcWithNullBytes(input);
    defer Allocator.free(result);

    // NFC of "hello" is still "hello"
    try std.testing.expectEqualStrings("hello", result);
}

test "tokenizer_apply_normalizer_spec sets normalization options" {
    var tok = ct.Tokenizer{
        .model = null,
        .type = .bpe,
        .normalizer = std.mem.zeroes(ct.Normalizer),
        .pretokenizer = std.mem.zeroes(ct.PreTokenizer),
        .postproc = std.mem.zeroes(ct.PostProcessor),
        .decoder = std.mem.zeroes(ct.Decoder),
        .padding = std.mem.zeroes(ct.Padding),
        .truncation = std.mem.zeroes(ct.Truncation),
        .added = null,
        .last_error = null,
    };

    const spec = ct.NormalizerSpec{
        .type = null,
        .lowercase = 1,
        .strip_accents = 1,
        .nfc = 1,
        .nfd = 0,
        .nfkc = 0,
        .clean_text = 1,
        .handle_chinese_chars = 0,
        .prepend = null,
        .replace_pattern = null,
        .replace_content = null,
    };

    tokenizer_apply_normalizer_spec(&tok, &spec);

    try std.testing.expectEqual(@as(c_int, 1), tok.normalizer.lowercase);
    try std.testing.expectEqual(@as(c_int, 1), tok.normalizer.strip_accents);
    try std.testing.expectEqual(@as(c_int, 1), tok.normalizer.nfc);
    try std.testing.expectEqual(@as(c_int, 0), tok.normalizer.nfd);
    try std.testing.expectEqual(@as(c_int, 1), tok.normalizer.clean_text);
}

test "stripAccents processes codepoint" {
    // stripAccents is a private function but we can test via normalize flow
    // This is a basic smoke test to ensure the function exists
    const codepoint: c.utf8proc_int32_t = 'e'; // Simple ASCII
    const result = stripAccents(codepoint);
    try std.testing.expectEqual(@as(c.utf8proc_int32_t, 'e'), result);
}

test "normalize_text requires integration testing" {
    // This function requires a fully initialized normalizer with:
    // - utf8proc library for Unicode normalization
    // - Full tokenizer context with normalizer spec
    // - Support for NFC/NFD/NFKC normalization
    // - Clean text and Chinese character handling
    // Integration tests: tests/tokenizer/test_*.py
}
