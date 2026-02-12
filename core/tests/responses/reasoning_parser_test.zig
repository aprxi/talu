//! Integration tests for responses.ReasoningParser
//!
//! ReasoningParser separates reasoning content (chain-of-thought) from
//! response content using configurable XML-style tag markers. Supports
//! streaming via processChunk and produces a ParseResult via finalize.

const std = @import("std");
const main = @import("main");

const ReasoningParser = main.responses.ReasoningParser;
const ReasoningFormat = main.responses.ReasoningFormat;

// ===== init =====

test "ReasoningParser: init creates markers from default tag" {
    var p = try ReasoningParser.init(std.testing.allocator, null);
    defer p.deinit();

    try std.testing.expectEqualStrings("<think>", p.start_marker);
    try std.testing.expectEqualStrings("</think>", p.end_marker);
    try std.testing.expectEqual(ReasoningFormat.none, p.format);
}

test "ReasoningParser: init creates markers from custom tag" {
    var p = try ReasoningParser.init(std.testing.allocator, "reasoning");
    defer p.deinit();

    try std.testing.expectEqualStrings("<reasoning>", p.start_marker);
    try std.testing.expectEqualStrings("</reasoning>", p.end_marker);
}

// ===== processChunk =====

test "ReasoningParser: processChunk separates reasoning from response" {
    var p = try ReasoningParser.init(std.testing.allocator, null);
    defer p.deinit();

    try p.processChunk("<think>step 1: analyze</think>The answer is 42.");
    const r = try p.finalize();

    try std.testing.expectEqualStrings("step 1: analyze", r.reasoning.?);
    try std.testing.expectEqualStrings("The answer is 42.", r.response.?);
    try std.testing.expectEqual(ReasoningFormat.xml_tags, p.format);
}

test "ReasoningParser: processChunk handles split across chunk boundaries" {
    var p = try ReasoningParser.init(std.testing.allocator, null);
    defer p.deinit();

    // Tag split: "<thi" | "nk>" and "</thi" | "nk>"
    try p.processChunk("hello<thi");
    try p.processChunk("nk>reasoning</thi");
    try p.processChunk("nk>world");
    const r = try p.finalize();

    try std.testing.expectEqualStrings("reasoning", r.reasoning.?);
    try std.testing.expectEqualStrings("helloworld", r.response.?);
}

test "ReasoningParser: processChunk byte-at-a-time streaming" {
    var p = try ReasoningParser.init(std.testing.allocator, null);
    defer p.deinit();

    const input = "<think>reason</think>answer";
    for (input) |byte| {
        try p.processChunk(&.{byte});
    }
    const r = try p.finalize();

    try std.testing.expectEqualStrings("reason", r.reasoning.?);
    try std.testing.expectEqualStrings("answer", r.response.?);
}

test "ReasoningParser: processChunk with no markers passes through as response" {
    var p = try ReasoningParser.init(std.testing.allocator, null);
    defer p.deinit();

    try p.processChunk("plain text without any tags");
    const r = try p.finalize();

    try std.testing.expect(r.reasoning == null);
    try std.testing.expectEqualStrings("plain text without any tags", r.response.?);
    try std.testing.expectEqual(ReasoningFormat.none, p.format);
}

test "ReasoningParser: processChunk accumulates multiple reasoning sections" {
    var p = try ReasoningParser.init(std.testing.allocator, null);
    defer p.deinit();

    try p.processChunk("<think>first</think>mid<think>second</think>end");
    const r = try p.finalize();

    try std.testing.expectEqualStrings("firstsecond", r.reasoning.?);
    try std.testing.expectEqualStrings("midend", r.response.?);
}

test "ReasoningParser: processChunk preserves orphan close tag as literal" {
    var p = try ReasoningParser.init(std.testing.allocator, null);
    defer p.deinit();

    try p.processChunk("text</think>more");
    const r = try p.finalize();

    try std.testing.expect(r.reasoning == null);
    try std.testing.expectEqualStrings("text</think>more", r.response.?);
}

test "ReasoningParser: processChunk with custom tag ignores default tag" {
    var p = try ReasoningParser.init(std.testing.allocator, "thought");
    defer p.deinit();

    try p.processChunk("<think>not detected</think>response");
    const r = try p.finalize();

    try std.testing.expect(r.reasoning == null);
    try std.testing.expectEqualStrings("<think>not detected</think>response", r.response.?);
}

// ===== finalize =====

test "ReasoningParser: finalize returns null for empty input" {
    var p = try ReasoningParser.init(std.testing.allocator, null);
    defer p.deinit();

    const r = try p.finalize();
    try std.testing.expect(r.reasoning == null);
    try std.testing.expect(r.response == null);
}

test "ReasoningParser: finalize flushes incomplete tag as literal text" {
    var p = try ReasoningParser.init(std.testing.allocator, null);
    defer p.deinit();

    try p.processChunk("hello<thin");
    const r = try p.finalize();

    try std.testing.expect(r.reasoning == null);
    try std.testing.expectEqualStrings("hello<thin", r.response.?);
}

test "ReasoningParser: finalize flushes unclosed reasoning as reasoning content" {
    var p = try ReasoningParser.init(std.testing.allocator, null);
    defer p.deinit();

    try p.processChunk("<think>reasoning without close");
    const r = try p.finalize();

    try std.testing.expectEqualStrings("reasoning without close", r.reasoning.?);
    try std.testing.expect(r.response == null);
}

// ===== post-tag newline swallowing =====

test "ReasoningParser: swallows one newline after end tag" {
    var p = try ReasoningParser.init(std.testing.allocator, null);
    defer p.deinit();

    try p.processChunk("<think>r\n</think>\nresponse");
    const r = try p.finalize();

    try std.testing.expectEqualStrings("r\n", r.reasoning.?);
    try std.testing.expectEqualStrings("response", r.response.?);
}

test "ReasoningParser: only first newline swallowed after end tag" {
    var p = try ReasoningParser.init(std.testing.allocator, null);
    defer p.deinit();

    try p.processChunk("<think>r</think>\n\nresponse");
    const r = try p.finalize();

    try std.testing.expectEqualStrings("r", r.reasoning.?);
    try std.testing.expectEqualStrings("\nresponse", r.response.?);
}

test "ReasoningParser: no newline after end tag preserves content" {
    var p = try ReasoningParser.init(std.testing.allocator, null);
    defer p.deinit();

    try p.processChunk("<think>r</think>response");
    const r = try p.finalize();

    try std.testing.expectEqualStrings("r", r.reasoning.?);
    try std.testing.expectEqualStrings("response", r.response.?);
}

// ===== deinit =====

test "ReasoningParser: deinit frees all buffers" {
    var p = try ReasoningParser.init(std.testing.allocator, null);

    // Exercise all internal buffers
    try p.processChunk("<think>reasoning</think>response");
    _ = try p.finalize();

    // deinit frees markers + all buffers; std.testing.allocator catches leaks.
    p.deinit();
}
