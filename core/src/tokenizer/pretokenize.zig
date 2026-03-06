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
extern fn pcre2_jit_compile_8(code: ?*anyopaque, options: u32) callconv(.c) c_int;

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
    // PCRE2_NO_UTF_CHECK: skip UTF-8 re-validation of the entire subject on each call.
    // The input was already validated when first received; re-checking on every match
    // in the loop is O(n) per call × O(n) calls = O(n²) total.
    return pcre2_match_8(code, subject, length, startoffset, PCRE2_NO_UTF_CHECK, match_data, null);
}

fn pcre2_get_ovector_pointer(match_data: ?*anyopaque) [*]c.PCRE2_SIZE {
    return pcre2_get_ovector_pointer_8(match_data);
}

fn pcre2_jit_compile(code: ?*anyopaque) void {
    _ = pcre2_jit_compile_8(code, PCRE2_JIT_COMPLETE);
}

// PCRE2 constants - can't translate macros
const PCRE2_ZERO_TERMINATED: c.PCRE2_SIZE = @bitCast(@as(isize, -1));
const PCRE2_UTF: u32 = 0x00080000;
const PCRE2_UCP: u32 = 0x00020000;
const PCRE2_NO_UTF_CHECK: u32 = 0x40000000;
const PCRE2_JIT_COMPLETE: u32 = 0x00000001;

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
    pcre2_jit_compile(regex_code);
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
    return try pretokenize_single_impl(pretokenizer, input, base_offset, false);
}

/// Internal implementation using error handling for cleaner code
fn pretokenize_single_impl(pretokenizer: ?*const ct.PreTokenizer, input: []const u8, base_offset: usize, skip_byte_level: bool) PretokenizeError!PretokenizeResult {
    var result = PretokenizeResult.init();
    result.zero_copy = skip_byte_level;
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
            try splitMetaspace(&result, input, base_offset, p.add_prefix_space != 0);
        } else if (p.re) |re| {
            if (isGpt2FastPath(p)) {
                try splitByGpt2Fast(&result, input, base_offset, re);
            } else {
                try splitByRegex(&result, input, base_offset, re, p.regex_split != 0, p.regex_invert != 0);
            }
        } else {
            try splitByWhitespace(&result, input, base_offset, p.whitespace != 0, p.punctuation != 0);
        }

        // Apply byte_level encoding if set.
        // skip_byte_level: BPE models with orig_byte_vocab_ids handle raw bytes
        // directly in encodeWordCore, so byte-level encoding is unnecessary.
        if (p.byte_level != 0 and !skip_byte_level) {
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

    // Pre-size token/range arrays to reduce growth reallocations
    const estimated_tokens = @max(input.len / 5, 16);
    try result.tokens.ensureTotalCapacity(Allocator, result.tokens.items.len + estimated_tokens);
    try result.ranges.ensureTotalCapacity(Allocator, result.ranges.items.len + estimated_tokens);

    // Determine what to emit:
    // - split=true,  invert=false: emit gaps only (MergedWith*, Removed)
    // - split=false, invert=false: emit matches AND gaps (Isolated)
    // - split=*,     invert=true:  emit matches only
    const emit_matches = !split_matches or invert_matches;
    const emit_gaps = !invert_matches;

    var cursor: usize = 0;
    while (cursor <= input.len) {
        const match_rc = pcre2_match(regex_code, @ptrCast(input.ptr), input.len, cursor, match_data);
        if (match_rc <= 0) {
            // No more matches - emit remaining text as a gap
            if (emit_gaps and cursor < input.len) {
                try appendToken(result, input[cursor..], base_offset + cursor);
            }
            break;
        }
        const ovector = pcre2_get_ovector_pointer(match_data);
        const match_start: usize = @intCast(ovector[0]);
        const match_end: usize = @intCast(ovector[1]);

        if (match_end == match_start) {
            cursor = match_end + 1;
            continue;
        }

        // Emit the gap before the match
        if (emit_gaps and match_start > cursor) {
            try appendToken(result, input[cursor..match_start], base_offset + cursor);
        }

        // Emit the match itself
        if (emit_matches) {
            try appendToken(result, input[match_start..match_end], base_offset + match_start);
        }

        cursor = match_end;
    }
}

// GPT-2 pattern constant for fast-path detection
const GPT2_PATTERN: [:0]const u8 = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

/// Check if pretokenizer is eligible for the fast GPT-2 path.
/// Requires the exact GPT-2 regex pattern with standard match mode (not split/invert).
fn isGpt2FastPath(p: *const ct.PreTokenizer) bool {
    if (p.regex_split != 0 or p.regex_invert != 0) return false;
    const pat = p.pattern orelse return false;
    const slice = std.mem.span(@as([*:0]const u8, @ptrCast(pat)));
    return std.mem.eql(u8, slice, GPT2_PATTERN);
}

/// Check if input[pos] starts a GPT-2 contraction suffix ('s, 't, 're, 've, 'm, 'll, 'd).
/// Returns the contraction length (2 or 3) or 0 if no match.
fn matchContraction(input: []const u8, pos: usize) usize {
    if (pos + 1 >= input.len) return 0;
    return switch (input[pos + 1]) {
        's', 't', 'm', 'd' => 2,
        'r' => if (pos + 2 < input.len and input[pos + 2] == 'e') 3 else 0,
        'v' => if (pos + 2 < input.len and input[pos + 2] == 'e') 3 else 0,
        'l' => if (pos + 2 < input.len and input[pos + 2] == 'l') 3 else 0,
        else => 0,
    };
}

/// Fast GPT-2 pretokenizer for ASCII text.
/// Replicates the GPT-2 regex behavior using direct byte dispatch.
/// Falls back to PCRE2 per-match for non-ASCII bytes.
fn splitByGpt2Fast(result: *PretokenizeResult, input: []const u8, base_offset: usize, regex_code: *anyopaque) !void {
    // Pre-size token/range arrays
    const estimated_tokens = @max(input.len / 5, 16);
    try result.tokens.ensureTotalCapacity(Allocator, result.tokens.items.len + estimated_tokens);
    try result.ranges.ensureTotalCapacity(Allocator, result.ranges.items.len + estimated_tokens);

    // Lazily create PCRE2 match data only when non-ASCII is encountered
    var match_data: ?*anyopaque = null;
    defer if (match_data) |md| pcre2_match_data_free(md);

    var cursor: usize = 0;
    while (cursor < input.len) {
        const ch = input[cursor];

        // Non-ASCII: fall back to PCRE2 for one match
        if (ch >= 0x80) {
            if (match_data == null) {
                match_data = pcre2_match_data_create_from_pattern(regex_code) orelse return error.OutOfMemory;
            }
            const rc = pcre2_match(regex_code, @ptrCast(input.ptr), input.len, cursor, match_data.?);
            if (rc > 0) {
                const ovector = pcre2_get_ovector_pointer(match_data.?);
                const match_start: usize = @intCast(ovector[0]);
                const match_end: usize = @intCast(ovector[1]);
                if (match_end > match_start) {
                    try appendToken(result, input[match_start..match_end], base_offset + match_start);
                    cursor = match_end;
                    continue;
                }
            }
            cursor += 1;
            continue;
        }

        // Alt 1: Contraction suffixes ('s, 't, 're, 've, 'm, 'll, 'd)
        if (ch == '\'') {
            const clen = matchContraction(input, cursor);
            if (clen > 0) {
                try appendToken(result, input[cursor .. cursor + clen], base_offset + cursor);
                cursor += clen;
                continue;
            }
            // Apostrophe without contraction: falls through to alt 4 (other chars)
        }

        // Alt 2/3/4 with optional leading space
        if (ch == ' ' and cursor + 1 < input.len and input[cursor + 1] < 0x80) {
            const next = input[cursor + 1];

            // Alt 2: Space + letters
            if (std.ascii.isAlphabetic(next)) {
                var end = cursor + 2;
                while (end < input.len and input[end] < 0x80 and std.ascii.isAlphabetic(input[end])) : (end += 1) {}
                try appendToken(result, input[cursor..end], base_offset + cursor);
                cursor = end;
                continue;
            }

            // Alt 3: Space + digits
            if (std.ascii.isDigit(next)) {
                var end = cursor + 2;
                while (end < input.len and input[end] < 0x80 and std.ascii.isDigit(input[end])) : (end += 1) {}
                try appendToken(result, input[cursor..end], base_offset + cursor);
                cursor = end;
                continue;
            }

            // Alt 4: Space + other (not ws/letter/digit)
            if (!std.ascii.isWhitespace(next) and !std.ascii.isAlphabetic(next) and !std.ascii.isDigit(next)) {
                var end = cursor + 2;
                while (end < input.len and input[end] < 0x80 and !std.ascii.isWhitespace(input[end]) and !std.ascii.isAlphabetic(input[end]) and !std.ascii.isDigit(input[end])) : (end += 1) {}
                try appendToken(result, input[cursor..end], base_offset + cursor);
                cursor = end;
                continue;
            }

            // Space followed by space/end: fall through to whitespace
        }

        // Alt 2: Letters (no leading space)
        if (std.ascii.isAlphabetic(ch)) {
            var end = cursor + 1;
            while (end < input.len and input[end] < 0x80 and std.ascii.isAlphabetic(input[end])) : (end += 1) {}
            try appendToken(result, input[cursor..end], base_offset + cursor);
            cursor = end;
            continue;
        }

        // Alt 3: Digits (no leading space)
        if (std.ascii.isDigit(ch)) {
            var end = cursor + 1;
            while (end < input.len and input[end] < 0x80 and std.ascii.isDigit(input[end])) : (end += 1) {}
            try appendToken(result, input[cursor..end], base_offset + cursor);
            cursor = end;
            continue;
        }

        // Alt 4: Other chars (not ws/letter/digit) — includes apostrophe without contraction
        if (!std.ascii.isWhitespace(ch)) {
            var end = cursor + 1;
            while (end < input.len and input[end] < 0x80 and !std.ascii.isWhitespace(input[end]) and !std.ascii.isAlphabetic(input[end]) and !std.ascii.isDigit(input[end])) : (end += 1) {}
            try appendToken(result, input[cursor..end], base_offset + cursor);
            cursor = end;
            continue;
        }

        // Alt 5/6: Whitespace run
        // \s+(?!\S)|\s+ semantics: if the run is 2+ chars and followed by
        // non-whitespace, leave the last whitespace char unconsumed so it
        // becomes the leading space of the next ` ?\p{L}+` etc. match.
        {
            var end = cursor + 1;
            while (end < input.len and input[end] < 0x80 and std.ascii.isWhitespace(input[end])) : (end += 1) {}
            if (end < input.len and end - cursor > 1) {
                end -= 1;
            }
            try appendToken(result, input[cursor..end], base_offset + cursor);
            cursor = end;
        }
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
fn splitMetaspace(result: *PretokenizeResult, input: []const u8, base_offset: usize, add_prefix_space: bool) !void {
    if (input.len == 0) return;

    // Replace spaces with ▁ (3 bytes per space instead of 1)
    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(Allocator);

    // Track original position for each byte in the replaced buffer
    var orig_pos_map = std.ArrayListUnmanaged(usize){};
    defer orig_pos_map.deinit(Allocator);

    // Prepend ▁ when add_prefix_space is set and input doesn't start with a space
    if (add_prefix_space and input[0] != ' ') {
        try buf.appendSlice(Allocator, "\xE2\x96\x81");
        // All 3 bytes of the prepended ▁ map to position 0 in the original
        try orig_pos_map.append(Allocator, 0);
        try orig_pos_map.append(Allocator, 0);
        try orig_pos_map.append(Allocator, 0);
    }

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
    if (result.zero_copy) {
        // Zero-copy: store pointer directly into input buffer (no arena alloc).
        try result.tokens.append(Allocator, .{ .ptr = @constCast(token_bytes.ptr), .len = token_bytes.len });
    } else {
        const arena_alloc = result.arena.allocator();
        const token_buf = arena_alloc.alloc(u8, token_bytes.len + 1) catch return error.OutOfMemory;
        @memcpy(token_buf[0..token_bytes.len], token_bytes);
        token_buf[token_bytes.len] = 0; // null terminator for C API
        try result.tokens.append(Allocator, .{ .ptr = token_buf.ptr, .len = token_bytes.len });
    }
    try result.ranges.append(Allocator, .{ .start = range_start, .end = range_start + token_bytes.len });
}

/// Append a single character token
fn appendTokenChar(result: *PretokenizeResult, byte_value: u8, position: usize) !void {
    const arena_alloc = result.arena.allocator();
    const char_buf = arena_alloc.alloc(u8, 2) catch return error.OutOfMemory;
    char_buf[0] = byte_value;
    char_buf[1] = 0;
    try result.tokens.append(Allocator, .{ .ptr = char_buf.ptr, .len = 1 });
    try result.ranges.append(Allocator, .{ .start = position, .end = position + 1 });
}

/// Apply GPT-2 byte-level encoding to all tokens.
/// Pre-computes encoded length to allocate exactly once per token,
/// avoiding the per-token ArrayList alloc/realloc/free cycle.
fn applyByteLevel(result: *PretokenizeResult) !void {
    const arena_alloc = result.arena.allocator();
    for (result.tokens.items) |*token_ptr| {
        const token_bytes = token_ptr.sliceConst();

        // Pre-compute encoded length (avoids ArrayList entirely)
        var encoded_len: usize = 0;
        for (token_bytes) |byte| {
            const cp = byteToUnicodeCodepoint(byte);
            encoded_len += if (cp < 0x80) @as(usize, 1) else if (cp < 0x800) @as(usize, 2) else @as(usize, 3);
        }

        const new_token = try arena_alloc.alloc(u8, encoded_len + 1);
        var pos: usize = 0;
        for (token_bytes) |byte| {
            var utf8_buf: [4]u8 = undefined;
            const len = utf8EncodeU32(byteToUnicodeCodepoint(byte), &utf8_buf);
            @memcpy(new_token[pos..][0..len], utf8_buf[0..len]);
            pos += len;
        }
        new_token[encoded_len] = 0;
        // Old token data stays in arena; freed when arena is deinited
        token_ptr.ptr = new_token.ptr;
        token_ptr.len = encoded_len;
    }
}

pub fn pretokenize(pretokenizer: ?*const ct.PreTokenizer, input: []const u8, input_range: Range, skip_byte_level: bool) PretokenizeError!PretokenizeResult {
    if (pretokenizer == null or pretokenizer.?.is_sequence == 0) {
        return pretokenize_single_impl(pretokenizer, input, input_range.start, skip_byte_level);
    }
    // Sequence pretokenizers: each step handles its own byte_level internally.
    // Never skip byte-level for sequence steps.
    return try pretokenize_sequence(pretokenizer.?, input, input_range);
}

/// Handle sequence pretokenizers
fn pretokenize_sequence(pretokenizer: *const ct.PreTokenizer, input: []const u8, input_range: Range) PretokenizeError!PretokenizeResult {
    var current = PretokenizeResult.init();
    errdefer current.deinit();

    // Start with input as single token
    try appendToken(&current, input, input_range.start);

    const seq_slice: [*]ct.PreTokenizer = @ptrCast(pretokenizer.seq.?);
    for (0..pretokenizer.seq_count) |sequence_index| {
        var next_result = PretokenizeResult.init();
        errdefer next_result.deinit();

        for (current.tokens.items, current.ranges.items) |token_item, token_range| {
            var segment_result = try pretokenize_single(&seq_slice[sequence_index], token_item.sliceConst(), token_range.start);
            defer segment_result.deinit();

            // Copy tokens to next_result's arena
            for (segment_result.tokens.items, segment_result.ranges.items) |segment_token, segment_range| {
                try appendToken(&next_result, segment_token.sliceConst(), segment_range.start);
            }
        }

        current.deinit();
        current = next_result;
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
    var result = PretokenizeResult.init();
    defer result.deinit();

    try splitByWhitespace(&result, "hello world", 0, true, false);

    try std.testing.expectEqual(@as(usize, 2), result.tokens.items.len);
    try std.testing.expectEqualStrings("hello", result.tokens.items[0].sliceConst());
    try std.testing.expectEqualStrings("world", result.tokens.items[1].sliceConst());
}

test "pretokenize splitByWhitespace punctuation" {
    var result = PretokenizeResult.init();
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
    var result = PretokenizeResult.init();
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
    var result = PretokenizeResult.init();
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

test "pretokenize requires integration testing 2" {
    // GPT-4 regex \p{N}{1,3} is greedy: "2025" → "202" + "5" in PCRE2.
    // HuggingFace fancy-regex produces "20" + "25" (possibly different semantics).
    // This difference is tracked as a known issue for granite-4.0-h-1B.
}

test "matchContraction matches valid suffixes" {
    try std.testing.expectEqual(@as(usize, 2), matchContraction("'s end", 0));
    try std.testing.expectEqual(@as(usize, 2), matchContraction("'t end", 0));
    try std.testing.expectEqual(@as(usize, 2), matchContraction("'m end", 0));
    try std.testing.expectEqual(@as(usize, 2), matchContraction("'d end", 0));
    try std.testing.expectEqual(@as(usize, 3), matchContraction("'re end", 0));
    try std.testing.expectEqual(@as(usize, 3), matchContraction("'ve end", 0));
    try std.testing.expectEqual(@as(usize, 3), matchContraction("'ll end", 0));
    try std.testing.expectEqual(@as(usize, 0), matchContraction("'x end", 0));
    try std.testing.expectEqual(@as(usize, 0), matchContraction("'", 0));
}

test "isGpt2FastPath detects pattern" {
    var pretok = std.mem.zeroes(ct.PreTokenizer);
    defer tokenizer_pretokenizer_free(&pretok);

    // Set GPT-2 pattern
    const rc = tokenizer_pretokenizer_set(&pretok, GPT2_PATTERN.ptr);
    try std.testing.expectEqual(@as(c_int, 0), rc);
    try std.testing.expect(isGpt2FastPath(&pretok));

    // Different split mode disqualifies
    pretok.regex_split = 1;
    try std.testing.expect(!isGpt2FastPath(&pretok));
    pretok.regex_split = 0;

    // Invert disqualifies
    pretok.regex_invert = 1;
    try std.testing.expect(!isGpt2FastPath(&pretok));
}

test "splitByGpt2Fast basic words" {
    var pretok = std.mem.zeroes(ct.PreTokenizer);
    defer tokenizer_pretokenizer_free(&pretok);
    _ = tokenizer_pretokenizer_set(&pretok, GPT2_PATTERN.ptr);

    var result = PretokenizeResult.init();
    defer result.deinit();
    try splitByGpt2Fast(&result, "Hello world", 0, pretok.re.?);

    try std.testing.expectEqual(@as(usize, 2), result.tokens.items.len);
    try std.testing.expectEqualStrings("Hello", result.tokens.items[0].sliceConst());
    try std.testing.expectEqualStrings(" world", result.tokens.items[1].sliceConst());
}

test "splitByGpt2Fast contractions" {
    var pretok = std.mem.zeroes(ct.PreTokenizer);
    defer tokenizer_pretokenizer_free(&pretok);
    _ = tokenizer_pretokenizer_set(&pretok, GPT2_PATTERN.ptr);

    var result = PretokenizeResult.init();
    defer result.deinit();
    try splitByGpt2Fast(&result, "don't", 0, pretok.re.?);

    try std.testing.expectEqual(@as(usize, 2), result.tokens.items.len);
    try std.testing.expectEqualStrings("don", result.tokens.items[0].sliceConst());
    try std.testing.expectEqualStrings("'t", result.tokens.items[1].sliceConst());
}

test "splitByGpt2Fast punctuation" {
    var pretok = std.mem.zeroes(ct.PreTokenizer);
    defer tokenizer_pretokenizer_free(&pretok);
    _ = tokenizer_pretokenizer_set(&pretok, GPT2_PATTERN.ptr);

    var result = PretokenizeResult.init();
    defer result.deinit();
    try splitByGpt2Fast(&result, "Hello, world!", 0, pretok.re.?);

    try std.testing.expectEqual(@as(usize, 4), result.tokens.items.len);
    try std.testing.expectEqualStrings("Hello", result.tokens.items[0].sliceConst());
    try std.testing.expectEqualStrings(",", result.tokens.items[1].sliceConst());
    try std.testing.expectEqualStrings(" world", result.tokens.items[2].sliceConst());
    try std.testing.expectEqualStrings("!", result.tokens.items[3].sliceConst());
}

test "splitByGpt2Fast numbers" {
    var pretok = std.mem.zeroes(ct.PreTokenizer);
    defer tokenizer_pretokenizer_free(&pretok);
    _ = tokenizer_pretokenizer_set(&pretok, GPT2_PATTERN.ptr);

    var result = PretokenizeResult.init();
    defer result.deinit();
    try splitByGpt2Fast(&result, "test 123", 0, pretok.re.?);

    try std.testing.expectEqual(@as(usize, 2), result.tokens.items.len);
    try std.testing.expectEqualStrings("test", result.tokens.items[0].sliceConst());
    try std.testing.expectEqualStrings(" 123", result.tokens.items[1].sliceConst());
}

test "splitByGpt2Fast whitespace leaves trailing space for next token" {
    var pretok = std.mem.zeroes(ct.PreTokenizer);
    defer tokenizer_pretokenizer_free(&pretok);
    _ = tokenizer_pretokenizer_set(&pretok, GPT2_PATTERN.ptr);

    var result = PretokenizeResult.init();
    defer result.deinit();
    try splitByGpt2Fast(&result, "a  b", 0, pretok.re.?);

    // \s+(?!\S) matches first space only (next char is also space),
    // then " b" matches ` ?\p{L}+`
    try std.testing.expectEqual(@as(usize, 3), result.tokens.items.len);
    try std.testing.expectEqualStrings("a", result.tokens.items[0].sliceConst());
    try std.testing.expectEqualStrings(" ", result.tokens.items[1].sliceConst());
    try std.testing.expectEqualStrings(" b", result.tokens.items[2].sliceConst());
}

test "splitByGpt2Fast mixed" {
    var pretok = std.mem.zeroes(ct.PreTokenizer);
    defer tokenizer_pretokenizer_free(&pretok);
    _ = tokenizer_pretokenizer_set(&pretok, GPT2_PATTERN.ptr);

    var result = PretokenizeResult.init();
    defer result.deinit();
    try splitByGpt2Fast(&result, "I'm 25!", 0, pretok.re.?);

    try std.testing.expectEqual(@as(usize, 4), result.tokens.items.len);
    try std.testing.expectEqualStrings("I", result.tokens.items[0].sliceConst());
    try std.testing.expectEqualStrings("'m", result.tokens.items[1].sliceConst());
    try std.testing.expectEqualStrings(" 25", result.tokens.items[2].sliceConst());
    try std.testing.expectEqualStrings("!", result.tokens.items[3].sliceConst());
}

test "splitByGpt2Fast matches splitByRegex on varied input" {
    var pretok = std.mem.zeroes(ct.PreTokenizer);
    defer tokenizer_pretokenizer_free(&pretok);
    _ = tokenizer_pretokenizer_set(&pretok, GPT2_PATTERN.ptr);

    const test_inputs = [_][]const u8{
        "Hello world",
        "don't won't I'm he'd they're we've she'll",
        "Hello, world! How's it going?",
        "test 123 foo456 789bar",
        "  hello  world  ",
        "\nhello\n\nworld\n",
        "a--b==c**d",
        " 's end",
        "word\t\tword",
        "'Tis the season",
        "price: $100.00!",
        "a  \t  b",
        "foo...bar!!!baz",
        " \n \n ",
        "  ",
        "x",
        "",
        "I'm 25! Don't stop. We're here--finally.",
    };

    for (test_inputs) |input| {
        var fast_result = PretokenizeResult.init();
        defer fast_result.deinit();
        try splitByGpt2Fast(&fast_result, input, 0, pretok.re.?);

        var regex_result = PretokenizeResult.init();
        defer regex_result.deinit();
        try splitByRegex(&regex_result, input, 0, pretok.re.?, false, false);

        // Compare token counts
        if (regex_result.tokens.items.len != fast_result.tokens.items.len) {
            std.debug.print("\nMISMATCH on input: \"{s}\" (len={d})\n", .{ input, input.len });
            std.debug.print("  regex tokens ({d}):", .{regex_result.tokens.items.len});
            for (regex_result.tokens.items) |tok| {
                std.debug.print(" \"{s}\"", .{tok.sliceConst()});
            }
            std.debug.print("\n  fast tokens ({d}):", .{fast_result.tokens.items.len});
            for (fast_result.tokens.items) |tok| {
                std.debug.print(" \"{s}\"", .{tok.sliceConst()});
            }
            std.debug.print("\n", .{});
            return error.TestUnexpectedResult;
        }

        // Compare each token
        for (fast_result.tokens.items, regex_result.tokens.items, 0..) |fast_tok, regex_tok, i| {
            if (!std.mem.eql(u8, fast_tok.sliceConst(), regex_tok.sliceConst())) {
                std.debug.print("\nTOKEN MISMATCH on input: \"{s}\" at index {d}\n", .{ input, i });
                std.debug.print("  regex: \"{s}\"\n  fast:  \"{s}\"\n", .{ regex_tok.sliceConst(), fast_tok.sliceConst() });
                return error.TestUnexpectedResult;
            }
        }
    }
}

test "tokenizer Split Removed overrides default regex pretokenizer for whitespace" {
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
    defer tokenizer_pretokenizer_free(&tok.pretokenizer);

    try std.testing.expectEqual(@as(c_int, 0), tokenizer_pretokenizer_set(&tok.pretokenizer, GPT2_PATTERN.ptr));

    const split_pattern: [:0]const u8 = "\\s+";
    const spec = ct.PreTokenizerSpec{
        .type = null,
        .add_prefix_space = 0,
        .trim_offsets = 1,
        .use_regex = 0,
        .byte_level = 0,
        .whitespace = 0,
        .punctuation = 0,
        .pattern = split_pattern.ptr,
        .regex_split = 1,
        .regex_invert = 0,
        .metaspace = 0,
    };
    tokenizer_apply_pretokenizer_spec(&tok, &spec);

    var result = try pretokenize(&tok.pretokenizer, "a b", .{ .start = 0, .end = 3 }, false);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 2), result.tokens.items.len);
    try std.testing.expectEqualStrings("a", result.tokens.items[0].sliceConst());
    try std.testing.expectEqualStrings("b", result.tokens.items[1].sliceConst());
}

test "tokenizer Split Removed overrides default regex pretokenizer for punctuation" {
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
    defer tokenizer_pretokenizer_free(&tok.pretokenizer);

    try std.testing.expectEqual(@as(c_int, 0), tokenizer_pretokenizer_set(&tok.pretokenizer, GPT2_PATTERN.ptr));

    const split_pattern: [:0]const u8 = "[,!.]+";
    const spec = ct.PreTokenizerSpec{
        .type = null,
        .add_prefix_space = 0,
        .trim_offsets = 1,
        .use_regex = 0,
        .byte_level = 0,
        .whitespace = 0,
        .punctuation = 0,
        .pattern = split_pattern.ptr,
        .regex_split = 1,
        .regex_invert = 0,
        .metaspace = 0,
    };
    tokenizer_apply_pretokenizer_spec(&tok, &spec);

    var result = try pretokenize(&tok.pretokenizer, "a,b!c.", .{ .start = 0, .end = 6 }, false);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 3), result.tokens.items.len);
    try std.testing.expectEqualStrings("a", result.tokens.items[0].sliceConst());
    try std.testing.expectEqualStrings("b", result.tokens.items[1].sliceConst());
    try std.testing.expectEqualStrings("c", result.tokens.items[2].sliceConst());
}
