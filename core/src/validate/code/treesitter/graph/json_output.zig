//! JSON serialization for call graph types.
//!
//! Converts CallableDefinitionInfo, CallSiteDetail, and AliasInfo into
//! JSON strings for the C API boundary.
//!
//! Thread safety: All functions are pure. Safe to call concurrently.

const std = @import("std");
const types = @import("types.zig");
const CallableDefinitionInfo = types.CallableDefinitionInfo;
const CallSiteDetail = types.CallSiteDetail;
const AliasInfo = types.AliasInfo;

/// Write a JSON-escaped string (without surrounding quotes) into buf.
fn writeEscaped(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), text: []const u8) !void {
    for (text) |ch| {
        switch (ch) {
            '"' => try buf.appendSlice(allocator, "\\\""),
            '\\' => try buf.appendSlice(allocator, "\\\\"),
            '\n' => try buf.appendSlice(allocator, "\\n"),
            '\r' => try buf.appendSlice(allocator, "\\r"),
            '\t' => try buf.appendSlice(allocator, "\\t"),
            else => try buf.append(allocator, ch),
        }
    }
}

/// Serialize callable definitions to a NUL-terminated JSON array string.
/// Caller owns the returned slice.
pub fn callablesToJson(
    allocator: std.mem.Allocator,
    callables: []const CallableDefinitionInfo,
) ![:0]u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    const w = buf.writer(allocator);

    try buf.append(allocator, '[');

    for (callables, 0..) |c, i| {
        if (i > 0) try buf.append(allocator, ',');

        try buf.appendSlice(allocator, "{\"fqn\":\"");
        try writeEscaped(allocator, &buf, c.fqn);
        try buf.appendSlice(allocator, "\",\"name_span\":");
        try writeSpan(w, c.name_span);
        try buf.appendSlice(allocator, ",\"body_span\":");
        try writeSpan(w, c.body_span);
        try buf.appendSlice(allocator, ",\"signature_span\":");
        try writeSpan(w, c.signature_span);
        try buf.appendSlice(allocator, ",\"language\":\"");
        try writeEscaped(allocator, &buf, c.language.name());
        try buf.appendSlice(allocator, "\",\"file_path\":\"");
        try writeEscaped(allocator, &buf, c.file_path);
        try buf.appendSlice(allocator, "\",\"visibility\":\"");
        try writeEscaped(allocator, &buf, c.visibility.name());

        // Parameters
        try buf.appendSlice(allocator, "\",\"parameters\":[");
        for (c.parameters, 0..) |param, pi| {
            if (pi > 0) try buf.append(allocator, ',');
            try buf.appendSlice(allocator, "{\"name\":\"");
            try writeEscaped(allocator, &buf, param.name);
            if (param.type_annotation) |ta| {
                try buf.appendSlice(allocator, "\",\"type\":\"");
                try writeEscaped(allocator, &buf, ta);
            }
            try buf.appendSlice(allocator, "\"}");
        }
        try buf.append(allocator, ']');

        // Return type
        if (c.return_type) |rt| {
            try buf.appendSlice(allocator, ",\"return_type\":\"");
            try writeEscaped(allocator, &buf, rt);
            try buf.append(allocator, '"');
        } else {
            try buf.appendSlice(allocator, ",\"return_type\":null");
        }

        try buf.append(allocator, '}');
    }

    try buf.append(allocator, ']');
    try buf.append(allocator, 0);

    const owned = try buf.toOwnedSlice(allocator);
    return owned[0 .. owned.len - 1 :0];
}

/// Serialize call sites to a NUL-terminated JSON array string.
/// Caller owns the returned slice.
pub fn callSitesToJson(
    allocator: std.mem.Allocator,
    call_sites: []const CallSiteDetail,
) ![:0]u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    const w = buf.writer(allocator);

    try buf.append(allocator, '[');

    for (call_sites, 0..) |cs, i| {
        if (i > 0) try buf.append(allocator, ',');

        try buf.appendSlice(allocator, "{\"raw_target_name\":\"");
        try writeEscaped(allocator, &buf, cs.raw_target_name);
        try buf.appendSlice(allocator, "\",\"call_expr_span\":");
        try writeSpan(w, cs.call_expr_span);
        try buf.appendSlice(allocator, ",\"target_name_span\":");
        try writeSpan(w, cs.target_name_span);
        try buf.appendSlice(allocator, ",\"definer_callable_fqn\":\"");
        try writeEscaped(allocator, &buf, cs.definer_callable_fqn);

        // Resolved paths
        try buf.appendSlice(allocator, "\",\"potential_resolved_paths\":[");
        for (cs.potential_resolved_paths, 0..) |rp, ri| {
            if (ri > 0) try buf.append(allocator, ',');
            try buf.append(allocator, '"');
            try writeEscaped(allocator, &buf, rp);
            try buf.append(allocator, '"');
        }
        try buf.append(allocator, ']');

        // Arguments
        try buf.appendSlice(allocator, ",\"arguments\":[");
        for (cs.arguments, 0..) |arg, ai| {
            if (ai > 0) try buf.append(allocator, ',');
            try buf.appendSlice(allocator, "{\"source\":\"");
            try writeEscaped(allocator, &buf, arg.source.name());
            try buf.appendSlice(allocator, "\",\"text\":\"");
            try writeEscaped(allocator, &buf, arg.text);
            try buf.appendSlice(allocator, "\"}");
        }
        try buf.append(allocator, ']');

        // Result variable
        if (cs.result_usage_variable) |rv| {
            try buf.appendSlice(allocator, ",\"result_usage_variable\":\"");
            try writeEscaped(allocator, &buf, rv);
            try buf.append(allocator, '"');
        } else {
            try buf.appendSlice(allocator, ",\"result_usage_variable\":null");
        }

        try buf.append(allocator, '}');
    }

    try buf.append(allocator, ']');
    try buf.append(allocator, 0);

    const owned = try buf.toOwnedSlice(allocator);
    return owned[0 .. owned.len - 1 :0];
}

/// Serialize aliases to a NUL-terminated JSON array string.
/// Caller owns the returned slice.
pub fn aliasesToJson(
    allocator: std.mem.Allocator,
    aliases: []const AliasInfo,
) ![:0]u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);

    try buf.append(allocator, '[');

    for (aliases, 0..) |a, i| {
        if (i > 0) try buf.append(allocator, ',');

        try buf.appendSlice(allocator, "{\"alias_fqn\":\"");
        try writeEscaped(allocator, &buf, a.alias_fqn);
        try buf.appendSlice(allocator, "\",\"target_path_guess\":\"");
        try writeEscaped(allocator, &buf, a.target_path_guess);
        try buf.appendSlice(allocator, "\",\"defining_module\":\"");
        try writeEscaped(allocator, &buf, a.defining_module);
        try buf.appendSlice(allocator, "\",\"is_public\":");
        try buf.appendSlice(allocator, if (a.is_public) "true" else "false");
        try buf.append(allocator, '}');
    }

    try buf.append(allocator, ']');
    try buf.append(allocator, 0);

    const owned = try buf.toOwnedSlice(allocator);
    return owned[0 .. owned.len - 1 :0];
}

fn writeSpan(w: anytype, span: types.Span) !void {
    try std.fmt.format(w, "{{\"start\":{d},\"end\":{d}}}", .{ span.start, span.end });
}

// =============================================================================
// Tests
// =============================================================================

test "callablesToJson produces valid JSON" {
    const callables = [_]CallableDefinitionInfo{
        .{
            .fqn = "::test::hello",
            .name_span = .{ .start = 4, .end = 9 },
            .body_span = .{ .start = 12, .end = 20 },
            .signature_span = .{ .start = 0, .end = 12 },
            .language = .python,
            .file_path = "test.py",
            .parameters = &.{},
            .return_type = null,
            .visibility = .public,
        },
    };

    const json = try callablesToJson(std.testing.allocator, &callables);
    defer std.testing.allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"fqn\":\"::test::hello\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"language\":\"python\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"visibility\":\"public\"") != null);
}

test "callSitesToJson produces valid JSON" {
    const calls = [_]CallSiteDetail{
        .{
            .raw_target_name = "foo",
            .potential_resolved_paths = &.{"::foo"},
            .call_expr_span = .{ .start = 0, .end = 6 },
            .target_name_span = .{ .start = 0, .end = 3 },
            .definer_callable_fqn = "::test::main",
            .arguments = &.{},
            .result_usage_variable = null,
        },
    };

    const json = try callSitesToJson(std.testing.allocator, &calls);
    defer std.testing.allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"raw_target_name\":\"foo\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"result_usage_variable\":null") != null);
}

test "callablesToJson handles empty array" {
    const json = try callablesToJson(std.testing.allocator, &.{});
    defer std.testing.allocator.free(json);
    try std.testing.expectEqualStrings("[]", json);
}

test "aliasesToJson produces valid JSON" {
    const aliases_data = [_]AliasInfo{
        .{
            .alias_fqn = "::mymod::np",
            .target_path_guess = "::numpy",
            .defining_module = "::mymod",
            .is_public = true,
        },
    };
    const json = try aliasesToJson(std.testing.allocator, &aliases_data);
    defer std.testing.allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"alias_fqn\":\"::mymod::np\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"target_path_guess\":\"::numpy\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"is_public\":true") != null);
}

test "aliasesToJson handles empty array" {
    const json = try aliasesToJson(std.testing.allocator, &.{});
    defer std.testing.allocator.free(json);
    try std.testing.expectEqualStrings("[]", json);
}
