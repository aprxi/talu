//! Pretokenization
//!
//! Splits input text into tokens before model encoding.
//! Supports regex-based splitting (GPT-2, BERT patterns), byte-level,
//! whitespace, and punctuation pretokenizers via PCRE2.

const std = @import("std");
const ct = @import("c_types.zig");
const utils = @import("utils.zig");
const types = @import("types.zig");
const strings = @import("strings.zig");
const log = @import("../log.zig");

const c = @cImport({
    @cDefine("PCRE2_CODE_UNIT_WIDTH", "8");
    @cInclude("pcre2.h");
});

const Allocator = types.Allocator;
const Range = types.Range;
const Token = types.Token;
const PretokenizeResult = types.PretokenizeResult;

pub const PretokenizeError = error{
    OutOfMemory,
};

// PCRE2 function-like macros don't translate, use _8 suffix directly
extern fn pcre2_compile_8(
    pattern: [*c]const u8,
    length: c.PCRE2_SIZE,
    options: u32,
    errorcode: *c_int,
    erroroffset: *c.PCRE2_SIZE,
    ccontext: ?*anyopaque,
) callconv(.c) ?*anyopaque;

extern fn pcre2_code_free_8(code: ?*anyopaque) callconv(.c) void;
extern fn pcre2_match_data_create_from_pattern_8(code: ?*anyopaque, gcontext: ?*anyopaque) callconv(.c) ?*anyopaque;
extern fn pcre2_match_data_free_8(match_data: ?*anyopaque) callconv(.c) void;
extern fn pcre2_match_8(
    code: ?*anyopaque,
    subject: [*c]const u8,
    length: c.PCRE2_SIZE,
    startoffset: c.PCRE2_SIZE,
    options: u32,
    match_data: ?*anyopaque,
    mcontext: ?*anyopaque,
) callconv(.c) c_int;
extern fn pcre2_get_ovector_pointer_8(match_data: ?*anyopaque) callconv(.c) [*]c.PCRE2_SIZE;

fn pcre2_compile(
    pattern: [*c]const u8,
    length: c.PCRE2_SIZE,
    options: u32,
    errorcode: *c_int,
    erroroffset: *c.PCRE2_SIZE,
    ccontext: ?*anyopaque,
) ?*anyopaque {
    return pcre2_compile_8(pattern, length, options, errorcode, erroroffset, ccontext);
}

fn pcre2_code_free(code: ?*anyopaque) void {
    pcre2_code_free_8(code);
}

fn pcre2_match_data_create_from_pattern(code: ?*anyopaque) ?*anyopaque {
    return pcre2_match_data_create_from_pattern_8(code, null);
}

fn pcre2_match_data_free(match_data: ?*anyopaque) void {
    pcre2_match_data_free_8(match_data);
}

fn pcre2_match(code: ?*anyopaque, subject: [*c]const u8, length: c.PCRE2_SIZE, startoffset: c.PCRE2_SIZE, match_data: ?*anyopaque) c_int {
    return pcre2_match_8(code, subject, length, startoffset, 0, match_data, null);
}

fn pcre2_get_ovector_pointer(match_data: ?*anyopaque) [*]c.PCRE2_SIZE {
    return pcre2_get_ovector_pointer_8(match_data);
}

// PCRE2 constants - can't translate macros
const PCRE2_ZERO_TERMINATED: c.PCRE2_SIZE = @bitCast(@as(isize, -1));
const PCRE2_UTF: u32 = 0x00080000;
const PCRE2_UCP: u32 = 0x00020000;

fn isPunctuation(ch: u8) bool {
    return switch (ch) {
        '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~' => true,
        else => false,
    };
}

pub fn tokenizer_pretokenizer_free(pretokenizer_opt: ?*ct.PreTokenizer) void {
    if (pretokenizer_opt == null) return;
    const pretokenizer = pretokenizer_opt.?;
    if (pretokenizer.seq) |sequence_ptr| {
        const sequence_slice: [*]ct.PreTokenizer = @ptrCast(sequence_ptr);
        var sequence_index: usize = 0;
        while (sequence_index < pretokenizer.seq_count) : (sequence_index += 1) {
            tokenizer_pretokenizer_free(&sequence_slice[sequence_index]);
        }
        Allocator.free(sequence_slice[0..pretokenizer.seq_count]);
        pretokenizer.seq = null;
        pretokenizer.seq_count = 0;
    }
    if (pretokenizer.re) |re| {
        pcre2_code_free(re);
        pretokenizer.re = null;
    }
    if (pretokenizer.pattern) |pat| {
        const slice = std.mem.span(@as([*:0]u8, @ptrCast(pat)));
        Allocator.free(slice);
        pretokenizer.pattern = null;
    }
}

pub fn tokenizer_pretokenizer_set(pretokenizer: *ct.PreTokenizer, pattern_opt: ?[*:0]const u8) c_int {
    tokenizer_pretokenizer_free(pretokenizer);
    if (pattern_opt == null) return 0;
    const pattern_ptr = pattern_opt.?;
    var error_code: c_int = 0;
    var error_offset: c.PCRE2_SIZE = 0;
    const regex_code = pcre2_compile(@ptrCast(pattern_ptr), PCRE2_ZERO_TERMINATED, PCRE2_UTF | PCRE2_UCP, &error_code, &error_offset, null);
    if (regex_code == null) return -1;
    const pattern_copy = strings.tokenizer_strdup(pattern_ptr) orelse {
        pcre2_code_free(regex_code);
        return -1;
    };
    pretokenizer.pattern = @ptrCast(pattern_copy);
    pretokenizer.re = @ptrCast(regex_code);
    return 0;
}

pub fn tokenizer_apply_pretokenizer_spec(tok: ?*ct.Tokenizer, spec: ?*const ct.PreTokenizerSpec) void {
    if (tok == null or spec == null) return;
    const tokenizer = tok.?;
    const pretokenizer_spec = spec.?;
    tokenizer.pretokenizer.add_prefix_space = pretokenizer_spec.add_prefix_space;
    tokenizer.pretokenizer.trim_offsets = pretokenizer_spec.trim_offsets;
    tokenizer.pretokenizer.byte_level = pretokenizer_spec.byte_level;
    tokenizer.pretokenizer.whitespace = pretokenizer_spec.whitespace;
    tokenizer.pretokenizer.punctuation = pretokenizer_spec.punctuation;
    tokenizer.pretokenizer.regex_split = pretokenizer_spec.regex_split;
    tokenizer.pretokenizer.regex_invert = pretokenizer_spec.regex_invert;
    tokenizer.pretokenizer.metaspace = pretokenizer_spec.metaspace;
    if (pretokenizer_spec.pattern) |pat| {
        _ = tokenizer_pretokenizer_set(&tokenizer.pretokenizer, @ptrCast(pat));
    } else if (pretokenizer_spec.whitespace != 0 or pretokenizer_spec.punctuation != 0 or pretokenizer_spec.metaspace != 0) {
        // Non-regex pretokenizer: clear any model-default regex so the
        // whitespace/punctuation/metaspace flags take effect.
        _ = tokenizer_pretokenizer_set(&tokenizer.pretokenizer, null);
    }
}

// Use shared utilities for byte-to-unicode and UTF-8 encoding
const byteToUnicodeCodepoint = utils.byteToUnicodeCodepoint;

fn utf8EncodeU32(cp: u32, out: *[4]u8) usize {
    // Wrapper to handle u32 codepoints (utils.utf8Encode takes i32)
    const len = utils.utf8Encode(@intCast(cp), out);
    return @intCast(len);
}

fn pretokenize_single(pretokenizer: ?*const ct.PreTokenizer, input: []const u8, base_offset: usize) PretokenizeError!PretokenizeResult {
    return try pretokenize_single_impl(pretokenizer, input, base_offset);
}

/// Internal implementation using error handling for cleaner code
fn pretokenize_single_impl(pretokenizer: ?*const ct.PreTokenizer, input: []const u8, base_offset: usize) PretokenizeError!PretokenizeResult {
    var result = PretokenizeResult{ .tokens = .{}, .ranges = .{} };
    errdefer result.deinit();

    if (pretokenizer) |p| {
        log.trace("tokenizer", "pretokenize", .{
            .input_len = input.len,
            .byte_level = p.byte_level,
            .regex_split = p.regex_split,
            .regex_invert = p.regex_invert,
        }, @src());

        if (p.metaspace != 0 and p.re == null) {
            // Metaspace: replace spaces with ▁, then split on word boundaries
            try splitMetaspace(&result, input, base_offset);
        } else if (p.re) |re| {
            try splitByRegex(&result, input, base_offset, re, p.regex_split != 0, p.regex_invert != 0);
        } else {
            try splitByWhitespace(&result, input, base_offset, p.whitespace != 0, p.punctuation != 0);
        }

        // Apply byte_level encoding if set
        if (p.byte_level != 0) {
            try applyByteLevel(&result);
        }
    } else {
        // No pretokenizer: split on whitespace by default
        try splitByWhitespace(&result, input, base_offset, true, false);
    }

    return result;
}

/// Split input by regex pattern
/// is_split: if true, split on pattern (emit gaps); if false, emit matches
/// is_invert: if true, emit matches instead of gaps (override split)
fn splitByRegex(result: *PretokenizeResult, input: []const u8, base_offset: usize, regex_code: *anyopaque, split_matches: bool, invert_matches: bool) !void {
    const match_data = pcre2_match_data_create_from_pattern(regex_code) orelse return error.OutOfMemory;
    defer pcre2_match_data_free(match_data);

    // Determine what to emit:
    // - is_split=true, is_invert=false: emit gaps (split on pattern)
    // - is_split=false, is_invert=false: emit matches
    // - is_split=true, is_invert=true: emit matches (invert overrides split)
    // - is_split=false, is_invert=true: emit matches (invert overrides emit gaps)
    const emit_gaps = split_matches and !invert_matches;

    var cursor: usize = 0;
    while (cursor <= input.len) {
        const match_rc = pcre2_match(regex_code, @ptrCast(input.ptr), input.len, cursor, match_data);
        if (match_rc <= 0) {
            // No more matches - emit remaining text if we're emitting gaps
            if (emit_gaps and cursor < input.len) {
                try appendToken(result, input[cursor..], base_offset + cursor);
            }
            break;
        }
        const ovector = pcre2_get_ovector_pointer(match_data);
        const match_start: usize = @intCast(ovector[0]);
        const match_end: usize = @intCast(ovector[1]);

        if (emit_gaps) {
            // Emit the gap before the match
            if (match_start > cursor) {
                try appendToken(result, input[cursor..match_start], base_offset + cursor);
            }
            cursor = if (match_end == match_start) match_end + 1 else match_end;
            continue;
        }

        // Emit matches
        if (match_end == match_start) {
            cursor = match_end + 1;
            continue;
        }

        log.trace("tokenizer", "Regex match", .{ .start = match_start, .end = match_end }, @src());
        try appendToken(result, input[match_start..match_end], base_offset + match_start);
        cursor = match_end;
    }
}

/// Split input by whitespace and optionally punctuation
fn splitByWhitespace(result: *PretokenizeResult, input: []const u8, base_offset: usize, split_whitespace: bool, split_punctuation: bool) !void {
    var token_start: usize = 0;
    var byte_index: usize = 0;
    while (byte_index < input.len) {
        const byte_value = input[byte_index];
        const is_space = split_whitespace and std.ascii.isWhitespace(byte_value);
        const is_punct = split_punctuation and isPunctuation(byte_value);

        if (is_space or is_punct) {
            if (byte_index > token_start) {
                try appendToken(result, input[token_start..byte_index], base_offset + token_start);
            }
            if (is_punct) {
                try appendTokenChar(result, byte_value, base_offset + byte_index);
            }
            token_start = byte_index + 1;
        }
        byte_index += 1;
    }
    if (byte_index > token_start) {
        try appendToken(result, input[token_start..byte_index], base_offset + token_start);
    }
}

/// Split input using Metaspace conventions.
/// Replaces spaces with ▁ (U+2581), then splits on word boundaries
/// (where ▁ is preceded by a non-▁ character). Consecutive ▁ chars
/// stay in one token. The resulting tokens have ▁ already embedded.
fn splitMetaspace(result: *PretokenizeResult, input: []const u8, base_offset: usize) !void {
    if (input.len == 0) return;

    // Replace spaces with ▁ (3 bytes per space instead of 1)
    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(Allocator);

    // Track original position for each byte in the replaced buffer
    var orig_pos_map = std.ArrayListUnmanaged(usize){};
    defer orig_pos_map.deinit(Allocator);

    for (input, 0..) |byte, orig_pos| {
        if (byte == ' ') {
            try buf.appendSlice(Allocator, "\xE2\x96\x81");
            try orig_pos_map.append(Allocator, orig_pos);
            try orig_pos_map.append(Allocator, orig_pos);
            try orig_pos_map.append(Allocator, orig_pos);
        } else {
            try buf.append(Allocator, byte);
            try orig_pos_map.append(Allocator, orig_pos);
        }
    }

    const replaced = buf.items;
    if (replaced.len == 0) return;

    // Split before each ▁ that is preceded by a non-▁ character (word boundary).
    // Consecutive ▁ chars form a single token. The first token includes leading ▁s.
    var token_start: usize = 0;
    var prev_was_sp = true; // don't split at start
    var i: usize = 0;
    while (i < replaced.len) {
        if (i + 2 < replaced.len and replaced[i] == 0xE2 and replaced[i + 1] == 0x96 and replaced[i + 2] == 0x81) {
            if (!prev_was_sp and i > token_start) {
                const orig_start = orig_pos_map.items[token_start];
                try appendToken(result, replaced[token_start..i], base_offset + orig_start);
                token_start = i;
            }
            prev_was_sp = true;
            i += 3;
        } else {
            prev_was_sp = false;
            i += 1;
        }
    }
    if (token_start < replaced.len) {
        const orig_start = orig_pos_map.items[token_start];
        try appendToken(result, replaced[token_start..], base_offset + orig_start);
    }
}

/// Append a token string to result
fn appendToken(result: *PretokenizeResult, token_bytes: []const u8, range_start: usize) !void {
    // Allocate with +1 for null terminator (C API convention)
    const token_buf = Allocator.alloc(u8, token_bytes.len + 1) catch return error.OutOfMemory;
    errdefer Allocator.free(token_buf);
    @memcpy(token_buf[0..token_bytes.len], token_bytes);
    token_buf[token_bytes.len] = 0; // null terminator for C API
    try result.tokens.append(Allocator, .{ .ptr = token_buf.ptr, .len = token_bytes.len });
    try result.ranges.append(Allocator, .{ .start = range_start, .end = range_start + token_bytes.len });
}

/// Append a single character token
fn appendTokenChar(result: *PretokenizeResult, byte_value: u8, position: usize) !void {
    const char_buf = try Allocator.alloc(u8, 2);
    char_buf[0] = byte_value;
    char_buf[1] = 0;
    errdefer Allocator.free(char_buf);
    try result.tokens.append(Allocator, .{ .ptr = char_buf.ptr, .len = 1 });
    try result.ranges.append(Allocator, .{ .start = position, .end = position + 1 });
}

/// Apply GPT-2 byte-level encoding to all tokens
fn applyByteLevel(result: *PretokenizeResult) !void {
    for (result.tokens.items) |*token_ptr| {
        const token_bytes = token_ptr.sliceConst();
        log.trace("tokenizer", "Byte-level encode", .{ .input_len = token_bytes.len }, @src());

        var encoded_bytes = std.ArrayListUnmanaged(u8){};
        defer encoded_bytes.deinit(Allocator);

        for (token_bytes) |byte| {
            const codepoint = byteToUnicodeCodepoint(byte);
            var utf8_buf: [4]u8 = undefined;
            const encoded_len = utf8EncodeU32(codepoint, &utf8_buf);
            try encoded_bytes.appendSlice(Allocator, utf8_buf[0..encoded_len]);
        }

        log.trace("tokenizer", "Byte-level result", .{ .output_len = encoded_bytes.items.len }, @src());

        const new_token = try Allocator.alloc(u8, encoded_bytes.items.len + 1);
        @memcpy(new_token[0..encoded_bytes.items.len], encoded_bytes.items);
        new_token[encoded_bytes.items.len] = 0;
        // Free old token data
        Allocator.free(token_ptr.ptr[0 .. token_ptr.len + 1]);
        // Update token
        token_ptr.ptr = new_token.ptr;
        token_ptr.len = encoded_bytes.items.len;
    }
}

pub fn pretokenize(pretokenizer: ?*const ct.PreTokenizer, input: []const u8, input_range: Range) PretokenizeError!PretokenizeResult {
    if (pretokenizer == null or pretokenizer.?.is_sequence == 0) {
        return pretokenize_single(pretokenizer, input, input_range.start);
    }
    return try pretokenize_sequence(pretokenizer.?, input, input_range);
}

/// Handle sequence pretokenizers
fn pretokenize_sequence(pretokenizer: *const ct.PreTokenizer, input: []const u8, input_range: Range) PretokenizeError!PretokenizeResult {
    var current = PretokenizeResult{ .tokens = .{}, .ranges = .{} };
    errdefer current.deinit();

    // Start with input as single token
    try appendToken(&current, input, input_range.start);

    const seq_slice: [*]ct.PreTokenizer = @ptrCast(pretokenizer.seq.?);
    for (0..pretokenizer.seq_count) |sequence_index| {
        var next_result = PretokenizeResult{ .tokens = .{}, .ranges = .{} };
        errdefer next_result.deinit();

        for (current.tokens.items, current.ranges.items) |token_item, token_range| {
            const token_bytes = token_item.sliceConst();
            var segment_result = try pretokenize_single(&seq_slice[sequence_index], token_bytes, token_range.start);

            // Transfer tokens (don't free them, just move to next)
            for (segment_result.tokens.items, segment_result.ranges.items) |segment_token, segment_range| {
                try next_result.tokens.append(Allocator, segment_token);
                try next_result.ranges.append(Allocator, segment_range);
            }
            // Just deinit containers, not the tokens themselves
            segment_result.tokens.deinit(Allocator);
            segment_result.ranges.deinit(Allocator);
        }

        current.deinit();
        current = next_result;
        next_result = .{ .tokens = .{}, .ranges = .{} }; // prevent errdefer from double-freeing
    }

    return current;
}

// =============================================================================
// Tests
// =============================================================================

// Note: pretokenize_single, pretokenize_sequence, and splitByRegex require
// full tokenizer context and PCRE2 regex engine. They are tested via
// integration tests in tests/tokenizer/.

test "isPunctuation recognizes punctuation chars" {
    try std.testing.expect(isPunctuation('!'));
    try std.testing.expect(isPunctuation('.'));
    try std.testing.expect(isPunctuation(','));
    try std.testing.expect(isPunctuation('?'));
    try std.testing.expect(isPunctuation(':'));
    try std.testing.expect(!isPunctuation('a'));
    try std.testing.expect(!isPunctuation('Z'));
    try std.testing.expect(!isPunctuation('5'));
    try std.testing.expect(!isPunctuation(' '));
}

test "pretokenize splitByWhitespace spaces" {
    var result = PretokenizeResult{ .tokens = .{}, .ranges = .{} };
    defer result.deinit();

    try splitByWhitespace(&result, "hello world", 0, true, false);

    try std.testing.expectEqual(@as(usize, 2), result.tokens.items.len);
    try std.testing.expectEqualStrings("hello", result.tokens.items[0].sliceConst());
    try std.testing.expectEqualStrings("world", result.tokens.items[1].sliceConst());
}

test "pretokenize splitByWhitespace punctuation" {
    var result = PretokenizeResult{ .tokens = .{}, .ranges = .{} };
    defer result.deinit();

    try splitByWhitespace(&result, "hello,world", 0, false, true);

    try std.testing.expectEqual(@as(usize, 3), result.tokens.items.len);
    try std.testing.expectEqualStrings("hello", result.tokens.items[0].sliceConst());
    try std.testing.expectEqualStrings(",", result.tokens.items[1].sliceConst());
    try std.testing.expectEqualStrings("world", result.tokens.items[2].sliceConst());
}

test "tokenizer_apply_pretokenizer_spec sets options" {
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

    const spec = ct.PreTokenizerSpec{
        .type = null,
        .add_prefix_space = 1,
        .trim_offsets = 1,
        .use_regex = 0,
        .byte_level = 0,
        .whitespace = 1,
        .punctuation = 1,
        .pattern = null,
        .regex_split = 0,
        .regex_invert = 0,
        .metaspace = 0,
    };

    tokenizer_apply_pretokenizer_spec(&tok, &spec);

    try std.testing.expectEqual(@as(c_int, 1), tok.pretokenizer.add_prefix_space);
    try std.testing.expectEqual(@as(c_int, 1), tok.pretokenizer.trim_offsets);
    try std.testing.expectEqual(@as(c_int, 1), tok.pretokenizer.whitespace);
    try std.testing.expectEqual(@as(c_int, 1), tok.pretokenizer.punctuation);
}

test "pretokenize appendToken allocates" {
    var result = PretokenizeResult{ .tokens = .{}, .ranges = .{} };
    defer result.deinit();

    try appendToken(&result, "hello", 0);
    try appendToken(&result, "world", 6);

    try std.testing.expectEqual(@as(usize, 2), result.tokens.items.len);
    try std.testing.expectEqualStrings("hello", result.tokens.items[0].sliceConst());
    try std.testing.expectEqualStrings("world", result.tokens.items[1].sliceConst());
    try std.testing.expectEqual(@as(usize, 0), result.ranges.items[0].start);
    try std.testing.expectEqual(@as(usize, 6), result.ranges.items[1].start);
}

test "pretokenize appendTokenChar single-char" {
    var result = PretokenizeResult{ .tokens = .{}, .ranges = .{} };
    defer result.deinit();

    try appendTokenChar(&result, ',', 5);

    try std.testing.expectEqual(@as(usize, 1), result.tokens.items.len);
    try std.testing.expectEqualStrings(",", result.tokens.items[0].sliceConst());
    try std.testing.expectEqual(@as(usize, 5), result.ranges.items[0].start);
    try std.testing.expectEqual(@as(usize, 6), result.ranges.items[0].end);
}

test "tokenizer_pretokenizer_free handles null pretokenizer" {
    tokenizer_pretokenizer_free(null);
    // Should not crash
}

test "tokenizer_pretokenizer_set compiles regex pattern" {
    var pretok = std.mem.zeroes(ct.PreTokenizer);
    defer tokenizer_pretokenizer_free(&pretok);

    const pattern: [:0]const u8 = "\\s+";
    const result = tokenizer_pretokenizer_set(&pretok, pattern.ptr);

    try std.testing.expectEqual(@as(c_int, 0), result);
    try std.testing.expect(pretok.re != null);
    try std.testing.expect(pretok.pattern != null);
}

test "tokenizer_pretokenizer_set handles null pattern" {
    var pretok = std.mem.zeroes(ct.PreTokenizer);
    const result = tokenizer_pretokenizer_set(&pretok, null);
    try std.testing.expectEqual(@as(c_int, 0), result);
}

test "pretokenize requires integration testing" {
    // This function requires a fully initialized pretokenizer with:
    // - Compiled PCRE2 regex patterns
    // - Proper byte-level encoding support
    // - Complete tokenizer context
    // Integration tests: tests/tokenizer/test_*.py
}
