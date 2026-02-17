//! Text search and filter helpers — all pure functions.
//!
//! Used by scan.zig for content matching and by root.zig for session filtering.
//! No dependencies on TableAdapter or any other session sub-module.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Case-insensitive ASCII substring search. Allocation-free.
///
/// Returns true if `needle` appears anywhere in `haystack` under case-insensitive
/// ASCII comparison. Bytes outside ASCII uppercase range (including UTF-8
/// continuation bytes >= 0x80) pass through unmodified — safe for UTF-8 input.
pub fn textContainsInsensitive(haystack: []const u8, needle: []const u8) bool {
    return textFindInsensitive(haystack, needle) != null;
}

/// Case-insensitive ASCII substring search. Returns the byte offset of the
/// first match, or null if not found.
pub fn textFindInsensitive(haystack: []const u8, needle: []const u8) ?usize {
    if (needle.len == 0) return 0;
    if (haystack.len < needle.len) return null;
    for (0..haystack.len - needle.len + 1) |i| {
        var is_match = true;
        for (0..needle.len) |j| {
            if (std.ascii.toLower(haystack[i + j]) != std.ascii.toLower(needle[j])) {
                is_match = false;
                break;
            }
        }
        if (is_match) return i;
    }
    return null;
}

/// Check if marker matches ANY of the markers in filter (OR logic).
/// Filter is space-separated. Exact match (case-sensitive).
pub fn markerMatchAny(marker: []const u8, filter: []const u8) bool {
    if (marker.len == 0) return false;
    var start: usize = 0;
    while (start < filter.len) {
        while (start < filter.len and filter[start] == ' ') : (start += 1) {}
        if (start >= filter.len) break;
        var end = start;
        while (end < filter.len and filter[end] != ' ') : (end += 1) {}
        const filter_marker = filter[start..end];
        if (filter_marker.len > 0 and std.mem.eql(u8, marker, filter_marker)) {
            return true;
        }
        start = end;
    }
    return false;
}

/// Check if model matches filter pattern (case-insensitive).
/// Supports wildcards: "qwen*" matches "qwen3-0.6b", "*llama*" matches "meta-llama-3".
pub fn modelMatchesFilter(model: []const u8, filter: []const u8) bool {
    if (filter.len == 0) return true;
    if (model.len == 0) return false;

    // Check for wildcards
    const has_prefix_wild = filter[0] == '*';
    const has_suffix_wild = filter[filter.len - 1] == '*';

    // Extract the pattern without wildcards
    const pattern_start: usize = if (has_prefix_wild) 1 else 0;
    const pattern_end: usize = if (has_suffix_wild and filter.len > 1) filter.len - 1 else filter.len;
    if (pattern_start >= pattern_end) return true; // Just wildcards

    const pattern = filter[pattern_start..pattern_end];

    if (has_prefix_wild and has_suffix_wild) {
        // *pattern* -> contains
        return textContainsInsensitive(model, pattern);
    } else if (has_prefix_wild) {
        // *pattern -> ends with
        if (model.len < pattern.len) return false;
        const suffix = model[model.len - pattern.len ..];
        return caseInsensitiveEqual(suffix, pattern);
    } else if (has_suffix_wild) {
        // pattern* -> starts with
        if (model.len < pattern.len) return false;
        const prefix = model[0..pattern.len];
        return caseInsensitiveEqual(prefix, pattern);
    } else {
        // No wildcards -> exact match (case-insensitive)
        return caseInsensitiveEqual(model, pattern);
    }
}

/// Case-insensitive equality check for ASCII strings.
pub fn caseInsensitiveEqual(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    for (a, b) |ca, cb| {
        if (std.ascii.toLower(ca) != std.ascii.toLower(cb)) return false;
    }
    return true;
}

/// Extract a ~200-byte snippet starting just before the first case-insensitive
/// match of `query` in `text`. Uses a small lead-in (~30 bytes) so the match
/// appears near the start of the snippet and won't be clipped by CSS line-clamp.
/// Caller owns the returned slice.
pub fn extractSnippet(text: []const u8, query: []const u8, alloc: Allocator) !?[]const u8 {
    const match_pos = textFindInsensitive(text, query) orelse return null;
    const lead_in = 30; // bytes of context before the match
    const snippet_len = 200;
    const start = if (match_pos > lead_in) match_pos - lead_in else 0;
    const end = @min(text.len, start + snippet_len);
    return try alloc.dupe(u8, text[start..end]);
}

/// Extract plain text from a JSON item payload by collecting all `"text":"..."`
/// string values. Handles basic JSON escape sequences (\", \\, \n, \t, \/).
/// Returns null if no text fields found. Caller owns the returned slice.
pub fn extractTextFromPayload(payload: []const u8, alloc: Allocator) !?[]const u8 {
    const marker = "\"text\":\"";
    var result = std.ArrayList(u8).empty;
    errdefer result.deinit(alloc);

    var pos: usize = 0;
    while (pos < payload.len) {
        // Find next "text":" marker
        const found = std.mem.indexOf(u8, payload[pos..], marker) orelse break;
        const str_start = pos + found + marker.len;
        if (str_start >= payload.len) break;

        // Separate text fields with a space
        if (result.items.len > 0) try result.append(alloc, ' ');

        // Read the JSON string value until unescaped closing quote
        var i = str_start;
        while (i < payload.len) {
            const c = payload[i];
            if (c == '"') break; // end of string
            if (c == '\\' and i + 1 < payload.len) {
                // Handle escape sequences
                const next = payload[i + 1];
                switch (next) {
                    '"', '\\', '/' => try result.append(alloc, next),
                    'n' => try result.append(alloc, '\n'),
                    't' => try result.append(alloc, '\t'),
                    'r' => try result.append(alloc, '\r'),
                    else => {
                        try result.append(alloc, c);
                        try result.append(alloc, next);
                    },
                }
                i += 2;
            } else {
                try result.append(alloc, c);
                i += 1;
            }
        }
        pos = if (i < payload.len) i + 1 else payload.len;
    }

    if (result.items.len == 0) {
        result.deinit(alloc);
        return null;
    }
    return try result.toOwnedSlice(alloc);
}

// =============================================================================
// Tests
// =============================================================================

test "textContainsInsensitive exact match" {
    try std.testing.expect(textContainsInsensitive("Rust", "Rust"));
    try std.testing.expect(textContainsInsensitive("hello", "hello"));
}

test "textContainsInsensitive case mismatch" {
    try std.testing.expect(textContainsInsensitive("Building a Rust CLI", "rust"));
    try std.testing.expect(textContainsInsensitive("Building a Rust CLI", "RUST"));
    try std.testing.expect(textContainsInsensitive("Building a Rust CLI", "rUsT"));
    try std.testing.expect(textContainsInsensitive("lowercase text", "LOWERCASE"));
}

test "textContainsInsensitive no match" {
    try std.testing.expect(!textContainsInsensitive("Building a Rust CLI", "python"));
    try std.testing.expect(!textContainsInsensitive("hello world", "xyz"));
}

test "textContainsInsensitive empty needle" {
    try std.testing.expect(textContainsInsensitive("anything", ""));
    try std.testing.expect(textContainsInsensitive("", ""));
}

test "textContainsInsensitive needle longer than haystack" {
    try std.testing.expect(!textContainsInsensitive("ab", "abcdef"));
    try std.testing.expect(!textContainsInsensitive("", "x"));
}

test "textContainsInsensitive substring at boundaries" {
    try std.testing.expect(textContainsInsensitive("hello world", "hello")); // start
    try std.testing.expect(textContainsInsensitive("hello world", "world")); // end
    try std.testing.expect(textContainsInsensitive("hello world", "lo wo")); // middle
}

test "textContainsInsensitive non-ascii bytes passthrough" {
    // UTF-8 multi-byte: bytes >= 0x80 are not corrupted by ASCII toLower.
    // "café" in UTF-8 = [99, 97, 102, 195, 169]
    try std.testing.expect(textContainsInsensitive("café", "café")); // exact byte match
    try std.testing.expect(textContainsInsensitive("café", "caf")); // ASCII prefix
    try std.testing.expect(!textContainsInsensitive("café", "CAFÉ")); // non-ASCII case: no match (expected)
}

// ---------------------------------------------------------------------------
// textFindInsensitive unit tests
// ---------------------------------------------------------------------------

test "textFindInsensitive returns position" {
    try std.testing.expectEqual(@as(?usize, 0), textFindInsensitive("hello world", "hello"));
    try std.testing.expectEqual(@as(?usize, 6), textFindInsensitive("hello world", "world"));
    try std.testing.expectEqual(@as(?usize, 3), textFindInsensitive("hello world", "lo wo"));
}

test "textFindInsensitive case insensitive" {
    try std.testing.expectEqual(@as(?usize, 0), textFindInsensitive("Hello World", "hello"));
    try std.testing.expectEqual(@as(?usize, 0), textFindInsensitive("hello world", "HELLO"));
}

test "textFindInsensitive no match returns null" {
    try std.testing.expectEqual(@as(?usize, null), textFindInsensitive("hello world", "xyz"));
}

test "textFindInsensitive empty needle returns zero" {
    try std.testing.expectEqual(@as(?usize, 0), textFindInsensitive("hello", ""));
}

test "textFindInsensitive needle longer returns null" {
    try std.testing.expectEqual(@as(?usize, null), textFindInsensitive("hi", "hello"));
}

// ---------------------------------------------------------------------------
// extractSnippet unit tests
// ---------------------------------------------------------------------------

test "extractSnippet basic centered window" {
    const alloc = std.testing.allocator;
    // Short payload (< 200 bytes): snippet is entire payload
    const snippet = (try extractSnippet("The quantum physics explanation", "quantum", alloc)).?;
    defer alloc.free(snippet);
    try std.testing.expectEqualStrings("The quantum physics explanation", snippet);
}

test "extractSnippet no match returns null" {
    const alloc = std.testing.allocator;
    try std.testing.expectEqual(@as(?[]const u8, null), try extractSnippet("hello world", "xyz", alloc));
}

test "extractSnippet long payload starts before match" {
    const alloc = std.testing.allocator;
    // Build a payload that's well over 200 bytes with the match in the middle
    const prefix = "A" ** 150;
    const suffix = "B" ** 150;
    const payload = prefix ++ "QUANTUM" ++ suffix;
    const snippet = (try extractSnippet(payload, "quantum", alloc)).?;
    defer alloc.free(snippet);

    // Snippet starts 30 bytes before match (pos 150), so start=120, end=min(307,320)=307 → 187
    try std.testing.expectEqual(@as(usize, 187), snippet.len);
    // Snippet must contain the match
    try std.testing.expect(textContainsInsensitive(snippet, "quantum"));
    // Match should appear near the start (within first 40 bytes)
    const match_pos = textFindInsensitive(snippet, "quantum").?;
    try std.testing.expect(match_pos <= 40);
}

test "extractSnippet match at very start" {
    const alloc = std.testing.allocator;
    const payload = "quantum" ++ "X" ** 300;
    const snippet = (try extractSnippet(payload, "quantum", alloc)).?;
    defer alloc.free(snippet);

    try std.testing.expectEqual(@as(usize, 200), snippet.len);
    // Should start at the beginning
    try std.testing.expect(std.mem.startsWith(u8, snippet, "quantum"));
}

test "extractSnippet match at very end" {
    const alloc = std.testing.allocator;
    const payload = "X" ** 300 ++ "quantum";
    const snippet = (try extractSnippet(payload, "quantum", alloc)).?;
    defer alloc.free(snippet);

    // lead_in=30: start = 300-30 = 270, end = min(307, 470) = 307 → 37 bytes
    try std.testing.expectEqual(@as(usize, 37), snippet.len);
    // Should end at the payload end
    try std.testing.expect(std.mem.endsWith(u8, snippet, "quantum"));
}

// ---------------------------------------------------------------------------
// extractTextFromPayload unit tests
// ---------------------------------------------------------------------------

test "extractTextFromPayload extracts text fields" {
    const alloc = std.testing.allocator;
    const payload =
        \\{"session_id":"s1","record":{"type":"message","content":[{"type":"output_text","text":"Hello world"}]}}
    ;
    const text = (try extractTextFromPayload(payload, alloc)).?;
    defer alloc.free(text);
    try std.testing.expectEqualStrings("Hello world", text);
}

test "extractTextFromPayload multiple text fields" {
    const alloc = std.testing.allocator;
    const payload =
        \\{"record":{"content":[{"type":"input_text","text":"question"},{"type":"output_text","text":"answer"}]}}
    ;
    const text = (try extractTextFromPayload(payload, alloc)).?;
    defer alloc.free(text);
    try std.testing.expectEqualStrings("question answer", text);
}

test "extractTextFromPayload handles escapes" {
    const alloc = std.testing.allocator;
    const payload =
        \\{"record":{"content":[{"type":"output_text","text":"line1\nline2"}]}}
    ;
    const text = (try extractTextFromPayload(payload, alloc)).?;
    defer alloc.free(text);
    try std.testing.expectEqualStrings("line1\nline2", text);
}

test "extractTextFromPayload no text fields returns null" {
    const alloc = std.testing.allocator;
    const payload =
        \\{"record":{"type":"function_call","name":"foo","arguments":"{}"}}
    ;
    try std.testing.expectEqual(@as(?[]const u8, null), try extractTextFromPayload(payload, alloc));
}

// ---------------------------------------------------------------------------
// markerMatchAny unit tests
// ---------------------------------------------------------------------------

test "markerMatchAny exact match in single-word filter" {
    try std.testing.expect(markerMatchAny("active", "active"));
}

test "markerMatchAny match among multiple filter words" {
    try std.testing.expect(markerMatchAny("archived", "active archived deleted"));
}

test "markerMatchAny no match" {
    try std.testing.expect(!markerMatchAny("pinned", "active archived deleted"));
}

test "markerMatchAny is case-sensitive" {
    try std.testing.expect(!markerMatchAny("Active", "active"));
    try std.testing.expect(!markerMatchAny("active", "Active"));
}

test "markerMatchAny empty marker returns false" {
    try std.testing.expect(!markerMatchAny("", "active archived"));
}

test "markerMatchAny empty filter returns false" {
    try std.testing.expect(!markerMatchAny("active", ""));
}

test "markerMatchAny partial word does not match" {
    try std.testing.expect(!markerMatchAny("act", "active archived"));
}

// ---------------------------------------------------------------------------
// modelMatchesFilter unit tests
// ---------------------------------------------------------------------------

test "modelMatchesFilter exact match case-insensitive" {
    try std.testing.expect(modelMatchesFilter("Qwen3-0.6B", "qwen3-0.6b"));
    try std.testing.expect(modelMatchesFilter("qwen3-0.6b", "Qwen3-0.6B"));
}

test "modelMatchesFilter prefix wildcard (suffix*)" {
    try std.testing.expect(modelMatchesFilter("qwen3-0.6b-gaf4", "qwen*"));
    try std.testing.expect(modelMatchesFilter("Qwen3-0.6B", "qwen*"));
}

test "modelMatchesFilter suffix wildcard (*suffix)" {
    try std.testing.expect(modelMatchesFilter("meta-llama-3", "*llama-3"));
    try std.testing.expect(!modelMatchesFilter("meta-llama-3", "*qwen"));
}

test "modelMatchesFilter contains wildcard (*pattern*)" {
    try std.testing.expect(modelMatchesFilter("meta-llama-3-instruct", "*llama*"));
    try std.testing.expect(!modelMatchesFilter("meta-llama-3", "*qwen*"));
}

test "modelMatchesFilter empty filter matches anything" {
    try std.testing.expect(modelMatchesFilter("any-model", ""));
}

test "modelMatchesFilter empty model matches nothing" {
    try std.testing.expect(!modelMatchesFilter("", "qwen*"));
}

test "modelMatchesFilter lone wildcard matches anything" {
    try std.testing.expect(modelMatchesFilter("any-model", "*"));
}

test "modelMatchesFilter no match without wildcard" {
    try std.testing.expect(!modelMatchesFilter("qwen3-0.6b", "llama"));
}

// ---------------------------------------------------------------------------
// caseInsensitiveEqual unit tests
// ---------------------------------------------------------------------------

test "caseInsensitiveEqual matching strings" {
    try std.testing.expect(caseInsensitiveEqual("Hello", "hello"));
    try std.testing.expect(caseInsensitiveEqual("ABC", "abc"));
    try std.testing.expect(caseInsensitiveEqual("test", "test"));
}

test "caseInsensitiveEqual different lengths" {
    try std.testing.expect(!caseInsensitiveEqual("abc", "abcd"));
    try std.testing.expect(!caseInsensitiveEqual("abcd", "abc"));
}

test "caseInsensitiveEqual non-matching same length" {
    try std.testing.expect(!caseInsensitiveEqual("abc", "xyz"));
}

test "caseInsensitiveEqual empty strings" {
    try std.testing.expect(caseInsensitiveEqual("", ""));
}
