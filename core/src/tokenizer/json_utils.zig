//! JSON Utilities
//!
//! JSON string unescaping for tokenizer vocab parsing.
//! Handles escape sequences (\n, \t, \uXXXX, etc.) in JSON strings.

const std = @import("std");

pub const JsonUnescapeError = error{
    OutOfMemory,
};

/// Unescape a JSON string.
/// Returns input unchanged if no escapes are present (zero-copy fast path).
pub fn unescapeJsonString(allocator: std.mem.Allocator, input: []const u8) JsonUnescapeError![]const u8 {
    var has_escape = false;
    for (input) |byte_value| {
        if (byte_value == '\\') {
            has_escape = true;
            break;
        }
    }
    if (!has_escape) return input;

    var output_bytes = std.ArrayListUnmanaged(u8){};
    errdefer output_bytes.deinit(allocator);

    var byte_index: usize = 0;
    while (byte_index < input.len) {
        if (input[byte_index] == '\\' and byte_index + 1 < input.len) {
            const escape = input[byte_index + 1];
            switch (escape) {
                'n' => {
                    try output_bytes.append(allocator, '\n');
                    byte_index += 2;
                },
                'r' => {
                    try output_bytes.append(allocator, '\r');
                    byte_index += 2;
                },
                't' => {
                    try output_bytes.append(allocator, '\t');
                    byte_index += 2;
                },
                '\\' => {
                    try output_bytes.append(allocator, '\\');
                    byte_index += 2;
                },
                '"' => {
                    try output_bytes.append(allocator, '"');
                    byte_index += 2;
                },
                '/' => {
                    try output_bytes.append(allocator, '/');
                    byte_index += 2;
                },
                'b' => {
                    try output_bytes.append(allocator, 0x08); // backspace
                    byte_index += 2;
                },
                'f' => {
                    try output_bytes.append(allocator, 0x0C); // form feed
                    byte_index += 2;
                },
                'u' => {
                    if (byte_index + 5 < input.len) {
                        const hex = input[byte_index + 2 .. byte_index + 6];
                        const codepoint = std.fmt.parseInt(u21, hex, 16) catch {
                            try output_bytes.append(allocator, input[byte_index]);
                            byte_index += 1;
                            continue;
                        };
                        var utf8_buf: [4]u8 = undefined;
                        const encoded_len = std.unicode.utf8Encode(codepoint, &utf8_buf) catch {
                            try output_bytes.append(allocator, input[byte_index]);
                            byte_index += 1;
                            continue;
                        };
                        try output_bytes.appendSlice(allocator, utf8_buf[0..encoded_len]);
                        byte_index += 6;
                    } else {
                        try output_bytes.append(allocator, input[byte_index]);
                        byte_index += 1;
                    }
                },
                else => {
                    // Unknown escape, keep as-is
                    try output_bytes.append(allocator, '\\');
                    try output_bytes.append(allocator, escape);
                    byte_index += 2;
                },
            }
        } else {
            try output_bytes.append(allocator, input[byte_index]);
            byte_index += 1;
        }
    }

    return try output_bytes.toOwnedSlice(allocator);
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

pub fn findJsonFieldValue(json_bytes: []const u8, field_bytes: []const u8) ?[]const u8 {
    const pos = findJsonKeyPos(json_bytes, field_bytes) orelse return null;
    var cursor = pos + field_bytes.len;
    skipJsonSeparators(json_bytes, &cursor);
    if (cursor >= json_bytes.len) return null;

    const value_start = cursor;
    while (cursor < json_bytes.len and
        json_bytes[cursor] != ',' and
        json_bytes[cursor] != '}' and
        json_bytes[cursor] != ']' and
        !std.ascii.isWhitespace(json_bytes[cursor])) : (cursor += 1)
    {}
    return json_bytes[value_start..cursor];
}

pub fn findJsonFieldString(json_bytes: []const u8, field_bytes: []const u8) ?[]const u8 {
    const pos = findJsonKeyPos(json_bytes, field_bytes) orelse return null;
    var cursor = pos + field_bytes.len;
    skipJsonSeparators(json_bytes, &cursor);
    if (cursor >= json_bytes.len or json_bytes[cursor] != '"') return null;

    cursor += 1;
    const value_start = cursor;
    while (cursor < json_bytes.len and json_bytes[cursor] != '"') {
        if (json_bytes[cursor] == '\\') cursor += 2 else cursor += 1;
    }
    if (cursor > json_bytes.len) return null;
    return json_bytes[value_start..cursor];
}

/// Parse a JSON field whose value is a ["string", number] tuple.
pub fn findJsonFieldArrayString(json_bytes: []const u8, field_bytes: []const u8) ?[]const u8 {
    const pos = findJsonKeyPos(json_bytes, field_bytes) orelse return null;
    var cursor = pos + field_bytes.len;
    skipJsonSeparators(json_bytes, &cursor);
    if (cursor >= json_bytes.len or json_bytes[cursor] != '[') return null;

    cursor += 1;
    skipJsonWhitespace(json_bytes, &cursor);
    if (cursor >= json_bytes.len or json_bytes[cursor] != '"') return null;

    cursor += 1;
    const value_start = cursor;
    while (cursor < json_bytes.len and json_bytes[cursor] != '"') {
        if (json_bytes[cursor] == '\\') cursor += 2 else cursor += 1;
    }
    if (cursor > json_bytes.len) return null;
    return json_bytes[value_start..cursor];
}

fn skipJsonSeparators(json_bytes: []const u8, cursor: *usize) void {
    while (cursor.* < json_bytes.len and
        (json_bytes[cursor.*] == ':' or std.ascii.isWhitespace(json_bytes[cursor.*]))) : (cursor.* += 1)
    {}
}

fn skipJsonWhitespace(json_bytes: []const u8, cursor: *usize) void {
    while (cursor.* < json_bytes.len and std.ascii.isWhitespace(json_bytes[cursor.*])) : (cursor.* += 1) {}
}

test "unescapeJsonString handles escapes and unicode" {
    const allocator = std.testing.allocator;
    const input = "a\\n\\u0041\\t\\\"b";
    const out = try unescapeJsonString(allocator, input);
    defer if (out.ptr != input.ptr) allocator.free(out);
    try std.testing.expect(std.mem.eql(u8, out, "a\nA\t\"b"));
}
