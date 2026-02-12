//! Integration tests for template.COutputSpanList
//!
//! COutputSpanList is a C-compatible struct that converts OutputSpan slices
//! into extern-compatible spans suitable for FFI. It owns copied variable paths.

const std = @import("std");
const main = @import("main");

const COutputSpanList = main.template.COutputSpanList;
const COutputSpan = main.template.COutputSpan;
const CSpanSourceType = main.template.CSpanSourceType;
const OutputSpan = main.template.OutputSpan;
const SpanSource = main.template.SpanSource;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "COutputSpanList type is accessible" {
    const T = COutputSpanList;
    _ = T;
}

test "COutputSpanList is a struct" {
    const info = @typeInfo(COutputSpanList);
    try std.testing.expect(info == .@"struct");
}

test "COutputSpanList has expected fields" {
    const info = @typeInfo(COutputSpanList);
    const fields = info.@"struct".fields;

    var has_spans = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "spans")) has_spans = true;
    }

    try std.testing.expect(has_spans);
}

// =============================================================================
// Method Tests
// =============================================================================

test "COutputSpanList has fromSpans method" {
    try std.testing.expect(@hasDecl(COutputSpanList, "fromSpans"));
}

test "COutputSpanList has deinit method" {
    try std.testing.expect(@hasDecl(COutputSpanList, "deinit"));
}

test "COutputSpanList has count method" {
    try std.testing.expect(@hasDecl(COutputSpanList, "count"));
}

// =============================================================================
// fromSpans Tests
// =============================================================================

test "COutputSpanList.fromSpans with empty slice" {
    const allocator = std.testing.allocator;
    const spans: []const OutputSpan = &.{};

    var list = try COutputSpanList.fromSpans(allocator, spans);
    defer list.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), list.count());
    try std.testing.expectEqual(@as(usize, 0), list.spans.len);
}

test "COutputSpanList.fromSpans with static text span" {
    const allocator = std.testing.allocator;
    const spans = [_]OutputSpan{
        .{ .start = 0, .end = 5, .source = .static_text },
    };

    var list = try COutputSpanList.fromSpans(allocator, &spans);
    defer list.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), list.count());
    try std.testing.expectEqual(@as(u32, 0), list.spans[0].start);
    try std.testing.expectEqual(@as(u32, 5), list.spans[0].end);
    try std.testing.expectEqual(CSpanSourceType.static_text, list.spans[0].source_type);
    try std.testing.expectEqual(@as(?[*:0]u8, null), list.spans[0].variable_path);
}

test "COutputSpanList.fromSpans with variable span" {
    const allocator = std.testing.allocator;
    const spans = [_]OutputSpan{
        .{ .start = 10, .end = 20, .source = .{ .variable = "user.name" } },
    };

    var list = try COutputSpanList.fromSpans(allocator, &spans);
    defer list.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), list.count());
    try std.testing.expectEqual(@as(u32, 10), list.spans[0].start);
    try std.testing.expectEqual(@as(u32, 20), list.spans[0].end);
    try std.testing.expectEqual(CSpanSourceType.variable, list.spans[0].source_type);
    try std.testing.expect(list.spans[0].variable_path != null);
    try std.testing.expectEqualStrings("user.name", std.mem.span(list.spans[0].variable_path.?));
}

test "COutputSpanList.fromSpans with expression span" {
    const allocator = std.testing.allocator;
    const spans = [_]OutputSpan{
        .{ .start = 5, .end = 15, .source = .expression },
    };

    var list = try COutputSpanList.fromSpans(allocator, &spans);
    defer list.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), list.count());
    try std.testing.expectEqual(CSpanSourceType.expression, list.spans[0].source_type);
    try std.testing.expectEqual(@as(?[*:0]u8, null), list.spans[0].variable_path);
}

test "COutputSpanList.fromSpans with multiple mixed spans" {
    const allocator = std.testing.allocator;
    const spans = [_]OutputSpan{
        .{ .start = 0, .end = 5, .source = .static_text },
        .{ .start = 5, .end = 10, .source = .{ .variable = "name" } },
        .{ .start = 10, .end = 15, .source = .expression },
    };

    var list = try COutputSpanList.fromSpans(allocator, &spans);
    defer list.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 3), list.count());

    // Static text
    try std.testing.expectEqual(CSpanSourceType.static_text, list.spans[0].source_type);
    try std.testing.expectEqual(@as(?[*:0]u8, null), list.spans[0].variable_path);

    // Variable
    try std.testing.expectEqual(CSpanSourceType.variable, list.spans[1].source_type);
    try std.testing.expectEqualStrings("name", std.mem.span(list.spans[1].variable_path.?));

    // Expression
    try std.testing.expectEqual(CSpanSourceType.expression, list.spans[2].source_type);
    try std.testing.expectEqual(@as(?[*:0]u8, null), list.spans[2].variable_path);
}

// =============================================================================
// deinit Tests
// =============================================================================

test "COutputSpanList.deinit clears spans" {
    const allocator = std.testing.allocator;
    const spans = [_]OutputSpan{
        .{ .start = 0, .end = 5, .source = .{ .variable = "test" } },
    };

    var list = try COutputSpanList.fromSpans(allocator, &spans);
    list.deinit(allocator);

    // After deinit, spans should be empty
    try std.testing.expectEqual(@as(usize, 0), list.spans.len);
}

// =============================================================================
// count Tests
// =============================================================================

test "COutputSpanList.count returns correct count" {
    const allocator = std.testing.allocator;
    const spans = [_]OutputSpan{
        .{ .start = 0, .end = 5, .source = .static_text },
        .{ .start = 5, .end = 10, .source = .static_text },
    };

    var list = try COutputSpanList.fromSpans(allocator, &spans);
    defer list.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), list.count());
}
