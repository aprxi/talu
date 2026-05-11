//! Added-token lookup and span matching for tokenizer encoding.

const std = @import("std");
const ct = @import("c_types.zig");
const types = @import("types.zig");
const utils = @import("utils.zig");
const c = @cImport({
    @cInclude("utf8proc.h");
});

const Normalized = types.Normalized;

pub const Span = struct {
    start: usize,
    end: usize,
    token: *const ct.AddedToken,
};

fn isWordByte(byte_value: u8) bool {
    return std.ascii.isAlphabetic(byte_value) or std.ascii.isDigit(byte_value) or byte_value == '_';
}

fn isWordCodepoint(codepoint: c.utf8proc_int32_t) bool {
    return switch (c.utf8proc_category(codepoint)) {
        c.UTF8PROC_CATEGORY_LU,
        c.UTF8PROC_CATEGORY_LL,
        c.UTF8PROC_CATEGORY_LT,
        c.UTF8PROC_CATEGORY_LM,
        c.UTF8PROC_CATEGORY_LO,
        c.UTF8PROC_CATEGORY_MN,
        c.UTF8PROC_CATEGORY_MC,
        c.UTF8PROC_CATEGORY_ME,
        c.UTF8PROC_CATEGORY_ND,
        c.UTF8PROC_CATEGORY_NL,
        c.UTF8PROC_CATEGORY_NO,
        => true,
        else => false,
    };
}

fn isWordAt(input_bytes: []const u8, position: usize) bool {
    if (position >= input_bytes.len) return false;
    const first_byte = input_bytes[position];
    if (first_byte < 0x80) return isWordByte(first_byte);

    var codepoint: c.utf8proc_int32_t = 0;
    const consumed = c.utf8proc_iterate(@ptrCast(input_bytes.ptr + position), @intCast(input_bytes.len - position), &codepoint);
    if (consumed <= 0) return false;
    return isWordCodepoint(codepoint);
}

fn isWordBefore(input_bytes: []const u8, position: usize) bool {
    if (position == 0) return false;

    var start = position - 1;
    while (start > 0 and utils.isUtf8ContinuationByte(input_bytes[start])) : (start -= 1) {}

    if (input_bytes[start] < 0x80) return isWordByte(input_bytes[start]);

    var codepoint: c.utf8proc_int32_t = 0;
    const consumed = c.utf8proc_iterate(@ptrCast(input_bytes.ptr + start), @intCast(input_bytes.len - start), &codepoint);
    if (consumed <= 0 or start + @as(usize, @intCast(consumed)) != position) return false;
    return isWordCodepoint(codepoint);
}

fn hasSingleWordBoundary(input_bytes: []const u8, position: usize, content_len: usize) bool {
    const left_boundary_ok = position == 0 or !isWordBefore(input_bytes, position);
    const right_pos = position + content_len;
    const right_boundary_ok = right_pos == input_bytes.len or !isWordAt(input_bytes, right_pos);
    return left_boundary_ok and right_boundary_ok;
}

fn consumeMatchWhitespaceForward(input_bytes: []const u8, start: usize) usize {
    var cursor = start;
    while (cursor < input_bytes.len) {
        const ws_len = utils.whitespaceLenAt(input_bytes, cursor);
        if (ws_len > 0) {
            cursor += ws_len;
            continue;
        }
        if (cursor + 3 <= input_bytes.len and
            input_bytes[cursor] == 0xE2 and
            input_bytes[cursor + 1] == 0x96 and
            input_bytes[cursor + 2] == 0x81)
        {
            cursor += 3;
            continue;
        }
        break;
    }
    return cursor;
}

pub fn leadingWhitespaceLen(input_bytes: []const u8) usize {
    var cursor: usize = 0;
    while (cursor < input_bytes.len) {
        const ws_len = utils.whitespaceLenAt(input_bytes, cursor);
        if (ws_len == 0) break;
        cursor += ws_len;
    }
    return cursor;
}

pub fn trailingWhitespaceStart(input_bytes: []const u8) usize {
    var cursor = input_bytes.len;
    while (cursor > 0) {
        var start = cursor - 1;
        while (start > 0 and utils.isUtf8ContinuationByte(input_bytes[start])) : (start -= 1) {}
        const ws_len = utils.whitespaceLenAt(input_bytes, start);
        if (ws_len == 0 or start + ws_len != cursor) break;
        cursor = start;
    }
    return cursor;
}

fn normalizedCursorForSourceEnd(normalized: *const Normalized, start_cursor: usize, source_end: usize) usize {
    if (source_end == 0) return start_cursor;
    var cursor = start_cursor;
    while (cursor < normalized.text.len and cursor < normalized.map_end.len) : (cursor += 1) {
        const mapped_end = normalized.map_end[cursor];
        if (mapped_end >= @as(i32, @intCast(source_end))) return cursor + 1;
    }
    return normalized.text.len;
}

fn matchesBoundaries(added_token: *const ct.AddedToken, input_bytes: []const u8, position: usize) bool {
    if (added_token.content == null) return false;
    const content = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(added_token.content.?)), 0);
    if (position + content.len > input_bytes.len) return false;
    if (!std.mem.eql(u8, input_bytes[position..][0..content.len], content)) return false;
    return added_token.single_word == 0 or hasSingleWordBoundary(input_bytes, position, content.len);
}

pub fn findContentById(tokenizer: *const ct.Tokenizer, id: i32) ?[]const u8 {
    var added_iter = tokenizer.added;
    while (added_iter) |added_token| : (added_iter = added_token.next) {
        if (added_token.id == id) {
            if (added_token.content) |content_ptr| {
                return std.mem.sliceTo(@as([*:0]const u8, @ptrCast(content_ptr)), 0);
            }
        }
    }
    return null;
}

pub fn findExact(tokenizer: *ct.Tokenizer, input: []const u8) ?*const ct.AddedToken {
    var added_iter = tokenizer.added;
    while (added_iter) |added_token| {
        if (added_token.content) |content_ptr| {
            const content = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(content_ptr)), 0);
            if (std.mem.eql(u8, content, input)) return added_token;
        }
        added_iter = added_token.next;
    }
    return null;
}

pub fn collectSpans(
    allocator: std.mem.Allocator,
    tokenizer: *ct.Tokenizer,
    normalized: *const Normalized,
    original_input: []const u8,
) ?std.ArrayListUnmanaged(Span) {
    if (tokenizer.added == null) return std.ArrayListUnmanaged(Span){};

    var spans = std.ArrayListUnmanaged(Span){};
    var cursor: usize = 0;
    while (cursor < normalized.text.len) {
        var best_match: ?Span = null;
        var best_span_len: usize = 0;

        var added_iter = tokenizer.added;
        while (added_iter) |added_token| {
            if (added_token.content == null) {
                added_iter = added_token.next;
                continue;
            }
            const content = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(added_token.content.?)), 0);
            if (content.len == 0) {
                added_iter = added_token.next;
                continue;
            }

            const text = if (added_token.normalized != 0) normalized.text else original_input;
            const text_pos_opt: ?usize = if (added_token.normalized != 0) cursor else blk: {
                if (cursor < normalized.map.len and normalized.map[cursor] >= 0) {
                    break :blk @as(usize, @intCast(normalized.map[cursor]));
                }
                break :blk null;
            };
            if (text_pos_opt == null) {
                added_iter = added_token.next;
                continue;
            }

            const text_pos = text_pos_opt.?;
            const content_pos = if (added_token.lstrip != 0) consumeMatchWhitespaceForward(text, text_pos) else text_pos;
            if (content_pos + content.len > text.len) {
                added_iter = added_token.next;
                continue;
            }
            if (!std.mem.eql(u8, text[content_pos..][0..content.len], content)) {
                added_iter = added_token.next;
                continue;
            }
            if (!matchesBoundaries(added_token, text, content_pos)) {
                added_iter = added_token.next;
                continue;
            }

            var text_end = content_pos + content.len;
            if (added_token.rstrip != 0) {
                text_end = consumeMatchWhitespaceForward(text, text_end);
            }

            const span_end = if (added_token.normalized != 0)
                cursor + (text_end - text_pos)
            else
                normalizedCursorForSourceEnd(normalized, cursor, text_end);

            const span = Span{ .start = cursor, .end = span_end, .token = added_token };
            const span_len = span.end - span.start;
            if (span_len > best_span_len) {
                best_match = span;
                best_span_len = span_len;
            }
            added_iter = added_token.next;
        }

        if (best_match) |matched| {
            spans.append(allocator, matched) catch {
                spans.deinit(allocator);
                return null;
            };
            cursor = matched.end;
        } else {
            cursor += 1;
        }
    }

    return spans;
}

test "findContentById returns added token content" {
    const content: [:0]const u8 = "extra";
    var added = ct.AddedToken{
        .content = @ptrCast(@constCast(content.ptr)),
        .id = 42,
        .special = 0,
        .single_word = 0,
        .lstrip = 0,
        .rstrip = 0,
        .normalized = 0,
        .next = null,
    };
    var tokenizer = std.mem.zeroes(ct.Tokenizer);
    tokenizer.added = &added;

    try std.testing.expectEqualStrings("extra", findContentById(&tokenizer, 42).?);
    try std.testing.expect(findContentById(&tokenizer, 7) == null);
}

test "findExact matches complete added token" {
    const content: [:0]const u8 = "whole";
    var added = ct.AddedToken{
        .content = @ptrCast(@constCast(content.ptr)),
        .id = 7,
        .special = 0,
        .single_word = 0,
        .lstrip = 0,
        .rstrip = 0,
        .normalized = 0,
        .next = null,
    };
    var tokenizer = std.mem.zeroes(ct.Tokenizer);
    tokenizer.added = &added;

    try std.testing.expect(findExact(&tokenizer, "whole") == &added);
    try std.testing.expect(findExact(&tokenizer, "whole!") == null);
}

test "leadingWhitespaceLen and trailingWhitespaceStart trim unicode whitespace" {
    const input = " \t\nabc \t";
    try std.testing.expectEqual(@as(usize, 3), leadingWhitespaceLen(input));
    try std.testing.expectEqual(@as(usize, 6), trailingWhitespaceStart(input));
}
