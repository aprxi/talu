//! Model-agnostic reasoning detection and separation.
//!
//! Separates reasoning content (e.g. chain-of-thought) from response content
//! in generated text. The parser detects XML-style tag pairs and routes bytes
//! to the appropriate buffer.
//!
//! Marker tag is configurable at init (default: "think").
//! Constructs "<tag>" / "</tag>" at runtime so the parser is not locked to any
//! single model family.
//!
//! Supported formats (detected from content):
//!   - XML-style: `<tag>...</tag>` (Qwen "think", DeepSeek-R1, etc.)
//!   - None: no markers detected → all content is response
//!
//! Matching is state-dependent: in normal state only the start marker is
//! recognized; in reasoning state only the end marker is recognized.  Tags
//! appearing in the "wrong" state are flushed as literal text.  This handles
//! nested `<think>` inside reasoning and orphan `</think>` outside reasoning
//! without special-casing.

const std = @import("std");

pub const ReasoningFormat = enum(u8) {
    /// No reasoning markers detected.
    none = 0,
    /// XML-style tag pair detected (e.g. `<think>...</think>`).
    xml_tags = 1,
};

pub const ReasoningState = enum(u8) {
    /// In response content (or before any reasoning section).
    normal = 0,
    /// Inside a reasoning section.
    reasoning = 1,
};

/// Result returned by `finalize`. Slices borrow from internal buffers and are
/// valid until `deinit` is called.
pub const ParseResult = struct {
    reasoning: ?[]const u8,
    response: ?[]const u8,
};

pub const ReasoningParser = struct {
    allocator: std.mem.Allocator,

    reasoning_buffer: std.ArrayListUnmanaged(u8) = .{},
    response_buffer: std.ArrayListUnmanaged(u8) = .{},

    /// Partial match buffer for token-boundary handling.
    ///
    /// Accumulates bytes that could be the prefix of the state-relevant
    /// marker.  When the buffer completes a marker the parser transitions
    /// state; when it can no longer match, the bytes are flushed as literal
    /// text to the active content buffer.
    partial_match_buf: std.ArrayListUnmanaged(u8) = .{},

    state: ReasoningState = .normal,
    format: ReasoningFormat = .none,

    /// When true, the next byte is swallowed if it is '\n'.
    /// Set after matching the end marker to avoid a leading blank line.
    swallow_next_newline: bool = false,

    /// Runtime marker strings, owned by this struct.
    start_marker: []const u8,
    end_marker: []const u8,

    const DEFAULT_TAG = "think";

    /// Initialize with optional tag name.  `null` defaults to `"think"`.
    pub fn init(allocator: std.mem.Allocator, tag_name: ?[]const u8) !ReasoningParser {
        const tag = tag_name orelse DEFAULT_TAG;
        const start = try std.fmt.allocPrint(allocator, "<{s}>", .{tag});
        errdefer allocator.free(start);
        const end = try std.fmt.allocPrint(allocator, "</{s}>", .{tag});
        return .{
            .allocator = allocator,
            .start_marker = start,
            .end_marker = end,
        };
    }

    /// Process a chunk of generated text.
    ///
    /// Bytes are scanned one at a time.  At each step the partial-match
    /// buffer is checked against the state-relevant marker:
    ///   1. Full match → state transition, buffer cleared, marker not emitted.
    ///   2. Prefix match → keep buffering.
    ///   3. No match → flush buffer as literal text to the active buffer.
    pub fn processChunk(self: *ReasoningParser, chunk: []const u8) !void {
        for (chunk) |byte| {
            // Swallow one newline immediately following end marker.
            if (self.swallow_next_newline) {
                self.swallow_next_newline = false;
                if (byte == '\n') continue;
            }

            try self.partial_match_buf.append(self.allocator, byte);

            const buf = self.partial_match_buf.items;

            // Complete marker check (state-dependent).
            if (self.state == .normal and std.mem.eql(u8, buf, self.start_marker)) {
                self.partial_match_buf.clearRetainingCapacity();
                self.state = .reasoning;
                self.format = .xml_tags;
                continue;
            }
            if (self.state == .reasoning and std.mem.eql(u8, buf, self.end_marker)) {
                self.partial_match_buf.clearRetainingCapacity();
                self.state = .normal;
                self.swallow_next_newline = true;
                continue;
            }

            // Prefix check against the relevant marker only.
            const is_prefix = switch (self.state) {
                .normal => std.mem.startsWith(u8, self.start_marker, buf),
                .reasoning => std.mem.startsWith(u8, self.end_marker, buf),
            };

            if (!is_prefix) {
                const target = if (self.state == .reasoning)
                    &self.reasoning_buffer
                else
                    &self.response_buffer;
                try target.appendSlice(self.allocator, buf);
                self.partial_match_buf.clearRetainingCapacity();
            }
        }
    }

    /// Flush remaining partial buffer and return separated content.
    /// Returned slices borrow from internal buffers — valid until `deinit`.
    pub fn finalize(self: *ReasoningParser) !ParseResult {
        if (self.partial_match_buf.items.len > 0) {
            const target = if (self.state == .reasoning)
                &self.reasoning_buffer
            else
                &self.response_buffer;
            try target.appendSlice(self.allocator, self.partial_match_buf.items);
            self.partial_match_buf.clearRetainingCapacity();
        }

        return .{
            .reasoning = if (self.reasoning_buffer.items.len > 0)
                self.reasoning_buffer.items
            else
                null,
            .response = if (self.response_buffer.items.len > 0)
                self.response_buffer.items
            else
                null,
        };
    }

    pub fn deinit(self: *ReasoningParser) void {
        self.allocator.free(self.start_marker);
        self.allocator.free(self.end_marker);
        self.reasoning_buffer.deinit(self.allocator);
        self.response_buffer.deinit(self.allocator);
        self.partial_match_buf.deinit(self.allocator);
    }
};

// =============================================================================
// Tests
// =============================================================================

test "ReasoningParser.init default tag" {
    const alloc = std.testing.allocator;
    var p = try ReasoningParser.init(alloc, null);
    defer p.deinit();
    try std.testing.expectEqualStrings("<think>", p.start_marker);
    try std.testing.expectEqualStrings("</think>", p.end_marker);
}

test "ReasoningParser.init custom tag" {
    const alloc = std.testing.allocator;
    var p = try ReasoningParser.init(alloc, "thought");
    defer p.deinit();
    try std.testing.expectEqualStrings("<thought>", p.start_marker);
    try std.testing.expectEqualStrings("</thought>", p.end_marker);
}

test "ReasoningParser.processChunk basic reasoning and response" {
    const alloc = std.testing.allocator;
    var p = try ReasoningParser.init(alloc, null);
    defer p.deinit();

    try p.processChunk("<think>reasoning</think>response");
    const r = try p.finalize();
    try std.testing.expectEqualStrings("reasoning", r.reasoning.?);
    try std.testing.expectEqualStrings("response", r.response.?);
    try std.testing.expectEqual(ReasoningFormat.xml_tags, p.format);
}

test "ReasoningParser.processChunk no tags passthrough" {
    const alloc = std.testing.allocator;
    var p = try ReasoningParser.init(alloc, null);
    defer p.deinit();

    try p.processChunk("just a normal response");
    const r = try p.finalize();
    try std.testing.expect(r.reasoning == null);
    try std.testing.expectEqualStrings("just a normal response", r.response.?);
    try std.testing.expectEqual(ReasoningFormat.none, p.format);
}

test "ReasoningParser.processChunk split across chunks" {
    const alloc = std.testing.allocator;
    var p = try ReasoningParser.init(alloc, null);
    defer p.deinit();

    try p.processChunk("text<thi");
    try p.processChunk("nk>reason</think>done");
    const r = try p.finalize();
    try std.testing.expectEqualStrings("reason", r.reasoning.?);
    try std.testing.expectEqualStrings("textdone", r.response.?);
}

test "ReasoningParser.processChunk unclosed tag" {
    const alloc = std.testing.allocator;
    var p = try ReasoningParser.init(alloc, null);
    defer p.deinit();

    try p.processChunk("<think>reasoning without close");
    const r = try p.finalize();
    try std.testing.expectEqualStrings("reasoning without close", r.reasoning.?);
    try std.testing.expect(r.response == null);
}

test "ReasoningParser.processChunk thinking-only with close" {
    const alloc = std.testing.allocator;
    var p = try ReasoningParser.init(alloc, null);
    defer p.deinit();

    try p.processChunk("<think>only reasoning</think>");
    const r = try p.finalize();
    try std.testing.expectEqualStrings("only reasoning", r.reasoning.?);
    try std.testing.expect(r.response == null);
}

test "ReasoningParser.processChunk orphan close tag preserved" {
    const alloc = std.testing.allocator;
    var p = try ReasoningParser.init(alloc, null);
    defer p.deinit();

    try p.processChunk("some text</think>more text");
    const r = try p.finalize();
    try std.testing.expect(r.reasoning == null);
    try std.testing.expectEqualStrings("some text</think>more text", r.response.?);
}

test "ReasoningParser.processChunk nested think tag preserved" {
    const alloc = std.testing.allocator;
    var p = try ReasoningParser.init(alloc, null);
    defer p.deinit();

    try p.processChunk("<think>outer <think>inner</think> still outer</think>response");
    const r = try p.finalize();
    try std.testing.expectEqualStrings("outer <think>inner", r.reasoning.?);
    try std.testing.expectEqualStrings(" still outer</think>response", r.response.?);
}

test "ReasoningParser.processChunk empty reasoning" {
    const alloc = std.testing.allocator;
    var p = try ReasoningParser.init(alloc, null);
    defer p.deinit();

    try p.processChunk("<think></think>response");
    const r = try p.finalize();
    try std.testing.expect(r.reasoning == null);
    try std.testing.expectEqualStrings("response", r.response.?);
}

test "ReasoningParser.processChunk multiple reasoning sections" {
    const alloc = std.testing.allocator;
    var p = try ReasoningParser.init(alloc, null);
    defer p.deinit();

    try p.processChunk("<think>first</think>middle<think>second</think>end");
    const r = try p.finalize();
    try std.testing.expectEqualStrings("firstsecond", r.reasoning.?);
    try std.testing.expectEqualStrings("middleend", r.response.?);
}

test "ReasoningParser.processChunk false prefix" {
    const alloc = std.testing.allocator;
    var p = try ReasoningParser.init(alloc, null);
    defer p.deinit();

    try p.processChunk("<thinking>not a tag");
    const r = try p.finalize();
    try std.testing.expect(r.reasoning == null);
    try std.testing.expectEqualStrings("<thinking>not a tag", r.response.?);
}

test "ReasoningParser.processChunk custom tag works" {
    const alloc = std.testing.allocator;
    var p = try ReasoningParser.init(alloc, "thought");
    defer p.deinit();

    try p.processChunk("<thought>reasoning</thought>response");
    const r = try p.finalize();
    try std.testing.expectEqualStrings("reasoning", r.reasoning.?);
    try std.testing.expectEqualStrings("response", r.response.?);
}

test "ReasoningParser.processChunk custom tag ignores default" {
    const alloc = std.testing.allocator;
    var p = try ReasoningParser.init(alloc, "thought");
    defer p.deinit();

    // <think> should NOT be recognized when tag is "thought"
    try p.processChunk("<think>not reasoning</think>response");
    const r = try p.finalize();
    try std.testing.expect(r.reasoning == null);
    try std.testing.expectEqualStrings("<think>not reasoning</think>response", r.response.?);
}

test "ReasoningParser.finalize empty input" {
    const alloc = std.testing.allocator;
    var p = try ReasoningParser.init(alloc, null);
    defer p.deinit();

    const r = try p.finalize();
    try std.testing.expect(r.reasoning == null);
    try std.testing.expect(r.response == null);
}

test "ReasoningParser.processChunk byte-at-a-time" {
    const alloc = std.testing.allocator;
    var p = try ReasoningParser.init(alloc, null);
    defer p.deinit();

    const input = "<think>r</think>d";
    for (input) |byte| {
        try p.processChunk(&.{byte});
    }
    const r = try p.finalize();
    try std.testing.expectEqualStrings("r", r.reasoning.?);
    try std.testing.expectEqualStrings("d", r.response.?);
}

test "ReasoningParser.processChunk incomplete start tag at end" {
    const alloc = std.testing.allocator;
    var p = try ReasoningParser.init(alloc, null);
    defer p.deinit();

    try p.processChunk("hello<thin");
    const r = try p.finalize();
    try std.testing.expect(r.reasoning == null);
    try std.testing.expectEqualStrings("hello<thin", r.response.?);
}

test "ReasoningParser.processChunk swallows newline after end tag" {
    const alloc = std.testing.allocator;
    var p = try ReasoningParser.init(alloc, null);
    defer p.deinit();

    try p.processChunk("<think>reasoning</think>\nresponse");
    const r = try p.finalize();
    try std.testing.expectEqualStrings("reasoning", r.reasoning.?);
    try std.testing.expectEqualStrings("response", r.response.?);
}

test "ReasoningParser.processChunk swallows newline across chunks" {
    const alloc = std.testing.allocator;
    var p = try ReasoningParser.init(alloc, null);
    defer p.deinit();

    try p.processChunk("<think>reasoning</think>");
    try p.processChunk("\nresponse");
    const r = try p.finalize();
    try std.testing.expectEqualStrings("reasoning", r.reasoning.?);
    try std.testing.expectEqualStrings("response", r.response.?);
}

test "ReasoningParser.processChunk preserves non-newline after end tag" {
    const alloc = std.testing.allocator;
    var p = try ReasoningParser.init(alloc, null);
    defer p.deinit();

    try p.processChunk("<think>reasoning</think>response");
    const r = try p.finalize();
    try std.testing.expectEqualStrings("reasoning", r.reasoning.?);
    try std.testing.expectEqualStrings("response", r.response.?);
}

test "ReasoningParser.processChunk swallows only one newline" {
    const alloc = std.testing.allocator;
    var p = try ReasoningParser.init(alloc, null);
    defer p.deinit();

    try p.processChunk("<think>reasoning</think>\n\nresponse");
    const r = try p.finalize();
    try std.testing.expectEqualStrings("reasoning", r.reasoning.?);
    try std.testing.expectEqualStrings("\nresponse", r.response.?);
}
