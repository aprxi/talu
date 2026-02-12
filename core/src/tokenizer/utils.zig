//! Shared text utilities for tokenization models
//!
//! Contains UTF-8 encoding/decoding and GPT-2 byte-to-unicode mapping.

const std = @import("std");

// =============================================================================
// UTF-8 Encoding/Decoding
// =============================================================================

/// Encode a Unicode code_point to UTF-8 bytes
/// Returns the number of bytes written (1-4)
pub fn utf8Encode(code_point: i32, out: *[4]u8) u8 {
    if (code_point < 0) return 0;
    const code_point_u21: u21 = @intCast(@min(code_point, 0x10FFFF));
    const byte_count = std.unicode.utf8Encode(code_point_u21, out) catch return 0;
    return @intCast(byte_count);
}

/// Decode a UTF-8 character from a slice at the given index
/// Advances idx by the character length, returns the code_point or -1 on error
pub fn utf8Decode(byte_values: []const u8, idx: *usize) i32 {
    if (idx.* >= byte_values.len) return -1;
    const byte_count = std.unicode.utf8ByteSequenceLength(byte_values[idx.*]) catch {
        idx.* += 1;
        return -1;
    };
    if (idx.* + byte_count > byte_values.len) {
        idx.* += 1;
        return -1;
    }
    const code_point = std.unicode.utf8Decode(byte_values[idx.* .. idx.* + byte_count]) catch {
        idx.* += 1;
        return -1;
    };
    idx.* += byte_count;
    return @intCast(code_point);
}

/// Get the byte length of a UTF-8 character from its first byte
pub fn utf8CharLen(first_byte: u8) usize {
    return std.unicode.utf8ByteSequenceLength(first_byte) catch 1;
}

/// GPT-2 byte-to-unicode mapping: converts a byte to its unicode code_point
/// Printable ASCII (33-126) maps to itself
/// Extended ASCII (161-172, 174-255) maps to itself
/// Other bytes (0-32, 127-160, 173) map to 256+
pub fn byteToUnicodeCodepoint(byte_value: u8) u32 {
    // Printable ASCII range (excluding space)
    if (byte_value >= 33 and byte_value <= 126) return @as(u32, byte_value);
    // Extended ASCII ranges
    if (byte_value >= 161 and byte_value <= 172) return @as(u32, byte_value);
    if (byte_value >= 174) return @as(u32, byte_value); // 174-255

    // Non-printable bytes map to 256+ in order of appearance
    // Order: 0-32, 127-160, 173
    var offset: u32 = 0;
    if (byte_value <= 32) {
        offset = byte_value;
    } else if (byte_value == 127) {
        offset = 33;
    } else if (byte_value >= 128 and byte_value <= 160) {
        offset = 34 + (byte_value - 128);
    } else if (byte_value == 173) {
        offset = 34 + 33;
    }
    return 256 + offset;
}

// =============================================================================
// JSON Parsing Utilities
// =============================================================================

/// Find a section in JSON by key, returns slice starting at the value (after colon)
pub fn findJsonSection(json: []const u8, key: []const u8) ?[]const u8 {
    var search_offset: usize = 0;
    while (std.mem.indexOfPos(u8, json, search_offset, key)) |pos| {
        var scan_offset = pos + key.len;
        // Skip whitespace and colon
        while (scan_offset < json.len and (json[scan_offset] == ' ' or json[scan_offset] == ':' or json[scan_offset] == '\t' or json[scan_offset] == '\n' or json[scan_offset] == '\r')) : (scan_offset += 1) {}
        if (scan_offset < json.len and (json[scan_offset] == '{' or json[scan_offset] == '[')) {
            return json[scan_offset..];
        }
        search_offset = pos + 1;
    }
    return null;
}

/// Find matching closing brace/bracket, handling nested structures and strings
pub fn findMatchingBrace(byte_values: []const u8, open: u8, close: u8) ?usize {
    var nesting_depth: usize = 0;
    var cursor: usize = 0;
    while (cursor < byte_values.len) {
        const current_byte = byte_values[cursor];
        if (current_byte == '"') {
            // Skip string content
            cursor += 1;
            while (cursor < byte_values.len) {
                if (byte_values[cursor] == '\\' and cursor + 1 < byte_values.len) {
                    cursor += 2; // skip escaped char
                } else if (byte_values[cursor] == '"') {
                    cursor += 1;
                    break;
                } else {
                    cursor += 1;
                }
            }
        } else {
            if (current_byte == open) nesting_depth += 1 else if (current_byte == close) {
                nesting_depth -= 1;
                if (nesting_depth == 0) return cursor + 1;
            }
            cursor += 1;
        }
    }
    return null;
}

// =============================================================================
// GPT-2 Byte-to-Unicode Mapping
// =============================================================================

/// GPT-2 style byte-to-unicode mapping table
/// Maps each byte (0-255) to a Unicode code_point that represents it
pub const ByteMapping = struct {
    /// Forward mapping: byte -> unicode code_point (as UTF-8 string)
    byte_to_unicode: [256][]const u8,
    /// Reverse mapping: unicode code_point -> original byte
    unicode_to_byte: [65536]i32,

    pub fn init(allocator: std.mem.Allocator) !ByteMapping {
        var self = ByteMapping{
            .byte_to_unicode = undefined,
            .unicode_to_byte = [_]i32{-1} ** 65536,
        };

        // Initialize byte_to_unicode
        for (&self.byte_to_unicode) |*slot| {
            slot.* = "";
        }

        // Build the GPT-2 byte mapping
        // Printable ASCII and extended ASCII are mapped to themselves
        // Control characters and other bytes are mapped to U+0100+
        var byte_values = [_]i32{0} ** 512;
        var codepoint_values = [_]i32{0} ** 512;
        var mapping_len: usize = 0;

        // Printable ASCII: 33-126
        for (33..127) |byte_value| {
            byte_values[mapping_len] = @intCast(byte_value);
            codepoint_values[mapping_len] = @intCast(byte_value);
            mapping_len += 1;
        }
        // Extended ASCII: 161-172, 174-255
        for (161..173) |byte_value| {
            byte_values[mapping_len] = @intCast(byte_value);
            codepoint_values[mapping_len] = @intCast(byte_value);
            mapping_len += 1;
        }
        for (174..256) |byte_value| {
            byte_values[mapping_len] = @intCast(byte_value);
            codepoint_values[mapping_len] = @intCast(byte_value);
            mapping_len += 1;
        }

        // Map remaining bytes (0-32, 127-160, 173) to U+0100+
        var offset_index: usize = 0;
        for (0..256) |byte_value| {
            var present = false;
            for (0..mapping_len) |map_idx| {
                if (byte_values[map_idx] == byte_value) {
                    present = true;
                    break;
                }
            }
            if (!present) {
                byte_values[mapping_len] = @intCast(byte_value);
                codepoint_values[mapping_len] = 256 + @as(i32, @intCast(offset_index));
                mapping_len += 1;
                offset_index += 1;
            }
        }

        // Encode each code_point as UTF-8 and store
        for (0..mapping_len) |map_idx| {
            const code_point = codepoint_values[map_idx];
            var utf8_buffer: [4]u8 = undefined;
            const utf8_len = utf8Encode(code_point, &utf8_buffer);
            if (utf8_len == 0) continue;

            const utf8_copy = try allocator.alloc(u8, @as(usize, utf8_len));
            @memcpy(utf8_copy, utf8_buffer[0..@as(usize, utf8_len)]);
            self.byte_to_unicode[@as(usize, @intCast(byte_values[map_idx]))] = utf8_copy;

            if (code_point >= 0 and code_point < self.unicode_to_byte.len) {
                self.unicode_to_byte[@as(usize, @intCast(code_point))] = byte_values[map_idx];
            }
        }

        return self;
    }

    pub fn deinit(self: *ByteMapping, allocator: std.mem.Allocator) void {
        for (self.byte_to_unicode) |unicode_bytes| {
            if (unicode_bytes.len > 0) allocator.free(unicode_bytes);
        }
    }
};

// =============================================================================
// Token Model Helpers
// =============================================================================

/// Set an unknown token in a fixed-size buffer (common pattern across models)
pub fn setUnkToken(unk_token: *[16]u8, token: []const u8) void {
    @memset(unk_token[0..], 0);
    const copy_len = @min(token.len, unk_token.len - 1);
    @memcpy(unk_token[0..copy_len], token[0..copy_len]);
    unk_token[copy_len] = 0;
}

/// Get a slice view of an unknown token buffer
pub fn unkSlice(unk_token: *const [16]u8) []const u8 {
    const c_string_ptr: [*:0]const u8 = @ptrCast(unk_token);
    return std.mem.sliceTo(c_string_ptr, 0);
}

/// Helper to get a typed model pointer from a tokenizer
fn getModel(comptime T: type, tok: anytype) ?*T {
    const model_pointer = tok.model orelse return null;
    return @ptrCast(@alignCast(model_pointer));
}

/// Helper to get a const typed model pointer from a tokenizer
fn getModelConst(comptime T: type, tok: anytype) ?*const T {
    const model_pointer = tok.model orelse return null;
    return @ptrCast(@alignCast(model_pointer));
}

// =============================================================================
// Tests
// =============================================================================

test "utf8Encode" {
    var utf8_buffer: [4]u8 = undefined;

    // ASCII
    try std.testing.expectEqual(@as(u8, 1), utf8Encode('A', &utf8_buffer));
    try std.testing.expectEqual(@as(u8, 'A'), utf8_buffer[0]);

    // 2-byte (ñ = U+00F1)
    try std.testing.expectEqual(@as(u8, 2), utf8Encode(0xF1, &utf8_buffer));

    // 3-byte (€ = U+20AC)
    try std.testing.expectEqual(@as(u8, 3), utf8Encode(0x20AC, &utf8_buffer));

    // 4-byte (𝄞 = U+1D11E)
    try std.testing.expectEqual(@as(u8, 4), utf8Encode(0x1D11E, &utf8_buffer));
}

test "utf8Decode" {
    const hello = "Hello";
    var idx: usize = 0;
    try std.testing.expectEqual(@as(i32, 'H'), utf8Decode(hello, &idx));
    try std.testing.expectEqual(@as(usize, 1), idx);

    const euro = "€"; // 3 bytes
    idx = 0;
    try std.testing.expectEqual(@as(i32, 0x20AC), utf8Decode(euro, &idx));
    try std.testing.expectEqual(@as(usize, 3), idx);
}

test "ByteMapping" {
    const allocator = std.testing.allocator;
    var mapping = try ByteMapping.init(allocator);
    defer mapping.deinit(allocator);

    // ASCII 'A' should map to itself
    try std.testing.expectEqualStrings("A", mapping.byte_to_unicode['A']);

    // Space (32) should map to U+0120 (Ġ)
    const space_unicode = mapping.byte_to_unicode[32];
    try std.testing.expect(space_unicode.len > 0);

    // Verify reverse mapping
    try std.testing.expectEqual(@as(i32, 'A'), mapping.unicode_to_byte['A']);
}

test "ByteMapping.init creates complete mapping" {
    const allocator = std.testing.allocator;
    var mapping = try ByteMapping.init(allocator);
    defer mapping.deinit(allocator);

    // All 256 bytes should have mappings
    for (0..256) |i| {
        try std.testing.expect(mapping.byte_to_unicode[i].len > 0);
    }
}

test "ByteMapping.deinit frees all allocated strings" {
    const allocator = std.testing.allocator;
    var mapping = try ByteMapping.init(allocator);
    // deinit frees all allocated UTF-8 strings
    mapping.deinit(allocator);
}

test "utf8CharLen - single byte ASCII" {
    try std.testing.expectEqual(@as(usize, 1), utf8CharLen('A'));
    try std.testing.expectEqual(@as(usize, 1), utf8CharLen('z'));
    try std.testing.expectEqual(@as(usize, 1), utf8CharLen('0'));
    try std.testing.expectEqual(@as(usize, 1), utf8CharLen(' '));
}

test "utf8CharLen - multi-byte characters" {
    // 2-byte character (ñ = U+00F1 = C3 B1)
    try std.testing.expectEqual(@as(usize, 2), utf8CharLen(0xC3));

    // 3-byte character (€ = U+20AC = E2 82 AC)
    try std.testing.expectEqual(@as(usize, 3), utf8CharLen(0xE2));

    // 4-byte character (𝄞 = U+1D11E = F0 9D 84 9E)
    try std.testing.expectEqual(@as(usize, 4), utf8CharLen(0xF0));
}

test "utf8CharLen - invalid UTF-8 sequences" {
    // Invalid UTF-8 start bytes should return 1 (fallback)
    try std.testing.expectEqual(@as(usize, 1), utf8CharLen(0xFF));
    try std.testing.expectEqual(@as(usize, 1), utf8CharLen(0xFE));
    try std.testing.expectEqual(@as(usize, 1), utf8CharLen(0x80)); // continuation byte
}

test "byteToUnicodeCodepoint - printable ASCII" {
    // Printable ASCII (33-126) maps to itself
    try std.testing.expectEqual(@as(u32, '!'), byteToUnicodeCodepoint('!'));
    try std.testing.expectEqual(@as(u32, 'A'), byteToUnicodeCodepoint('A'));
    try std.testing.expectEqual(@as(u32, 'z'), byteToUnicodeCodepoint('z'));
    try std.testing.expectEqual(@as(u32, '~'), byteToUnicodeCodepoint('~'));
}

test "byteToUnicodeCodepoint - extended ASCII" {
    // Extended ASCII (161-172, 174-255) maps to itself
    try std.testing.expectEqual(@as(u32, 161), byteToUnicodeCodepoint(161));
    try std.testing.expectEqual(@as(u32, 172), byteToUnicodeCodepoint(172));
    try std.testing.expectEqual(@as(u32, 174), byteToUnicodeCodepoint(174));
    try std.testing.expectEqual(@as(u32, 255), byteToUnicodeCodepoint(255));
}

test "byteToUnicodeCodepoint - control characters" {
    // Control characters (0-32) map to 256+
    try std.testing.expectEqual(@as(u32, 256), byteToUnicodeCodepoint(0)); // NUL
    try std.testing.expectEqual(@as(u32, 256 + 10), byteToUnicodeCodepoint(10)); // LF
    try std.testing.expectEqual(@as(u32, 256 + 32), byteToUnicodeCodepoint(32)); // space
}

test "byteToUnicodeCodepoint - special bytes" {
    // DEL (127) maps to 256 + 33
    try std.testing.expectEqual(@as(u32, 256 + 33), byteToUnicodeCodepoint(127));

    // High control chars (128-160) map to 256 + 34+
    try std.testing.expectEqual(@as(u32, 256 + 34), byteToUnicodeCodepoint(128));
    try std.testing.expectEqual(@as(u32, 256 + 34 + 32), byteToUnicodeCodepoint(160));

    // Soft hyphen (173) maps to 256 + 67
    try std.testing.expectEqual(@as(u32, 256 + 67), byteToUnicodeCodepoint(173));
}

test "findJsonSection - simple object" {
    const json = "{\"key1\": {\"nested\": true}, \"key2\": [1, 2, 3]}";

    // Find key1 section
    const section1 = findJsonSection(json, "\"key1\"");
    try std.testing.expect(section1 != null);
    try std.testing.expect(section1.?[0] == '{');

    // Find key2 section
    const section2 = findJsonSection(json, "\"key2\"");
    try std.testing.expect(section2 != null);
    try std.testing.expect(section2.?[0] == '[');
}

test "findJsonSection - with whitespace" {
    const json = "{\"key\"  :  \n  {\"value\": 42}}";
    const section = findJsonSection(json, "\"key\"");
    try std.testing.expect(section != null);
    try std.testing.expect(section.?[0] == '{');
}

test "findJsonSection - nested objects" {
    const json = "{\"outer\": {\"inner\": {\"deep\": [1, 2, 3]}}}";

    // Find outer
    const outer = findJsonSection(json, "\"outer\"");
    try std.testing.expect(outer != null);
    try std.testing.expect(outer.?[0] == '{');

    // Find inner
    const inner = findJsonSection(json, "\"inner\"");
    try std.testing.expect(inner != null);
    try std.testing.expect(inner.?[0] == '{');

    // Find deep
    const deep = findJsonSection(json, "\"deep\"");
    try std.testing.expect(deep != null);
    try std.testing.expect(deep.?[0] == '[');
}

test "findJsonSection - key not found" {
    const json = "{\"key1\": {}, \"key2\": []}";
    const section = findJsonSection(json, "\"key3\"");
    try std.testing.expect(section == null);
}

test "findJsonSection - key exists but not followed by brace/bracket" {
    const json = "{\"key\": \"value\"}";
    const section = findJsonSection(json, "\"key\"");
    // Should return null since the value is a string, not object/array
    try std.testing.expect(section == null);
}

test "findMatchingBrace - simple object" {
    const json = "{\"key\": \"value\"}";
    const end = findMatchingBrace(json, '{', '}');
    try std.testing.expect(end != null);
    try std.testing.expectEqual(@as(usize, json.len), end.?);
}

test "findMatchingBrace - simple array" {
    const json = "[1, 2, 3]";
    const end = findMatchingBrace(json, '[', ']');
    try std.testing.expect(end != null);
    try std.testing.expectEqual(@as(usize, json.len), end.?);
}

test "findMatchingBrace - nested objects" {
    const json = "{\"a\": {\"b\": {\"c\": 1}}}";
    const end = findMatchingBrace(json, '{', '}');
    try std.testing.expect(end != null);
    try std.testing.expectEqual(@as(usize, json.len), end.?);
}

test "findMatchingBrace - nested arrays" {
    const json = "[[1, 2], [3, 4], [5, [6, 7]]]";
    const end = findMatchingBrace(json, '[', ']');
    try std.testing.expect(end != null);
    try std.testing.expectEqual(@as(usize, json.len), end.?);
}

test "findMatchingBrace - mixed nesting" {
    const json = "{\"array\": [1, {\"nested\": [2, 3]}], \"obj\": {}}";
    const end = findMatchingBrace(json, '{', '}');
    try std.testing.expect(end != null);
    try std.testing.expectEqual(@as(usize, json.len), end.?);
}

test "findMatchingBrace - strings with braces" {
    // Braces/brackets inside strings should be ignored
    const json = "{\"key\": \"value with { and } and [ and ]\"}";
    const end = findMatchingBrace(json, '{', '}');
    try std.testing.expect(end != null);
    try std.testing.expectEqual(@as(usize, json.len), end.?);
}

test "findMatchingBrace - strings with escaped quotes" {
    const json = "{\"key\": \"value \\\" with escaped quote\"}";
    const end = findMatchingBrace(json, '{', '}');
    try std.testing.expect(end != null);
    try std.testing.expectEqual(@as(usize, json.len), end.?);
}

test "findMatchingBrace - complex escaped strings" {
    const json = "{\"a\": \"\\\"\\\\{\", \"b\": \"}\\\"]\"}";
    const end = findMatchingBrace(json, '{', '}');
    try std.testing.expect(end != null);
    try std.testing.expectEqual(@as(usize, json.len), end.?);
}

test "findMatchingBrace - unmatched brace" {
    const json = "{\"key\": {\"nested\": 1}";
    const end = findMatchingBrace(json, '{', '}');
    try std.testing.expect(end == null);
}

test "findMatchingBrace - unmatched array" {
    const json = "[1, 2, [3, 4]";
    const end = findMatchingBrace(json, '[', ']');
    try std.testing.expect(end == null);
}

test "findMatchingBrace - empty structures" {
    const empty_obj = "{}";
    const end_obj = findMatchingBrace(empty_obj, '{', '}');
    try std.testing.expect(end_obj != null);
    try std.testing.expectEqual(@as(usize, 2), end_obj.?);

    const empty_arr = "[]";
    const end_arr = findMatchingBrace(empty_arr, '[', ']');
    try std.testing.expect(end_arr != null);
    try std.testing.expectEqual(@as(usize, 2), end_arr.?);
}

test "setUnkToken - normal token" {
    var unk_token: [16]u8 = undefined;
    setUnkToken(&unk_token, "<unk>");

    try std.testing.expectEqualStrings("<unk>", unkSlice(&unk_token));
}

test "setUnkToken - empty token" {
    var unk_token: [16]u8 = undefined;
    setUnkToken(&unk_token, "");

    try std.testing.expectEqualStrings("", unkSlice(&unk_token));
}

test "setUnkToken - long token truncation" {
    var unk_token: [16]u8 = undefined;
    const long_token = "this_is_a_very_long_token_that_exceeds_buffer";
    setUnkToken(&unk_token, long_token);

    const result = unkSlice(&unk_token);
    // Should be truncated to 15 chars max (buffer size - 1 for null terminator)
    try std.testing.expect(result.len <= 15);
    try std.testing.expectEqualStrings("this_is_a_very_", result);
}

test "setUnkToken - exactly buffer size" {
    var unk_token: [16]u8 = undefined;
    const exact_token = "exactly_15_char";
    setUnkToken(&unk_token, exact_token);

    try std.testing.expectEqualStrings("exactly_15_char", unkSlice(&unk_token));
}

test "setUnkToken - multiple sets" {
    var unk_token: [16]u8 = undefined;

    setUnkToken(&unk_token, "first");
    try std.testing.expectEqualStrings("first", unkSlice(&unk_token));

    setUnkToken(&unk_token, "second");
    try std.testing.expectEqualStrings("second", unkSlice(&unk_token));

    // Set with shorter token - should clear previous data
    setUnkToken(&unk_token, "new");
    try std.testing.expectEqualStrings("new", unkSlice(&unk_token));
}

test "unkSlice - zero length" {
    var unk_token: [16]u8 = undefined;
    @memset(&unk_token, 0);

    try std.testing.expectEqualStrings("", unkSlice(&unk_token));
}

test "unkSlice - partial buffer" {
    var unk_token: [16]u8 = undefined;
    @memset(&unk_token, 0);
    @memcpy(unk_token[0..5], "hello");

    try std.testing.expectEqualStrings("hello", unkSlice(&unk_token));
}
