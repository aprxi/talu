//! Template C API
//!
//! C-callable functions for Jinja2 template rendering.
//! Delegates to src/template/ for all template logic.

const std = @import("std");
const template_engine = @import("../template/root.zig");
const capi_error = @import("error.zig");
const error_codes = @import("error_codes.zig");
const gen_config_mod = @import("../inference/config/generation.zig");

/// C allocator for FFI lifetime management.
const allocator = std.heap.c_allocator;

// =============================================================================
// Render Functions
// =============================================================================

/// Render a Jinja2 template with JSON variables.
/// Set strict=true to raise errors on undefined variables.
/// Caller must free result with talu_text_free.
pub export fn talu_template_render(
    template_str: [*:0]const u8,
    json_vars: [*:0]const u8,
    strict: bool,
    out_rendered: *?[*:0]u8,
) callconv(.c) i32 {
    capi_error.clearError();
    out_rendered.* = null;

    var result = template_engine.renderFromJson(
        allocator,
        std.mem.span(template_str),
        std.mem.span(json_vars),
        strict,
    );
    defer result.deinit(allocator);

    // Handle JSON parse error
    if (result.error_message) |msg| {
        capi_error.setErrorWithCode(.template_invalid_json, "Template JSON parse failed: {s}", .{msg});
        return @intFromEnum(error_codes.ErrorCode.template_invalid_json);
    }

    // Handle render error - context already set by template module
    if (result.err) |err| {
        capi_error.setError(err, "Template render failed", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    }

    // Success - copy to C string
    return copyToCString(result.output.?, out_rendered);
}

/// Render with custom Python filters.
/// Caller must free result with talu_text_free.
pub export fn talu_template_render_with_filters(
    template_str: [*:0]const u8,
    json_vars: [*:0]const u8,
    strict: bool,
    filter_names: [*]const [*:0]const u8,
    filter_callbacks: [*]const template_engine.CustomFilterCallback,
    filter_user_data: [*]const ?*anyopaque,
    num_filters: usize,
    out_rendered: *?[*:0]u8,
) callconv(.c) i32 {
    capi_error.clearError();
    out_rendered.* = null;

    // Build custom filter set
    var custom_filters = template_engine.CustomFilterSet.init();
    defer custom_filters.deinit(allocator);

    for (0..num_filters) |i| {
        custom_filters.put(allocator, std.mem.span(filter_names[i]), .{
            .callback = filter_callbacks[i],
            .user_data = filter_user_data[i],
        }) catch {
            capi_error.setErrorWithCode(.out_of_memory, "OutOfMemory: failed to register custom filter", .{});
            return @intFromEnum(error_codes.ErrorCode.out_of_memory);
        };
    }

    var result = template_engine.renderFromJsonWithFilters(
        allocator,
        std.mem.span(template_str),
        std.mem.span(json_vars),
        strict,
        if (num_filters > 0) &custom_filters else null,
    );
    defer result.deinit(allocator);

    if (result.error_message) |msg| {
        capi_error.setErrorWithCode(.template_invalid_json, "Template JSON parse failed: {s}", .{msg});
        return @intFromEnum(error_codes.ErrorCode.template_invalid_json);
    }

    // Handle render error - context already set by template module
    if (result.err) |err| {
        capi_error.setError(err, "Template render failed", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    }

    return copyToCString(result.output.?, out_rendered);
}

// =============================================================================
// Chat Template Source
// =============================================================================

/// Get raw chat template source from a model directory.
/// Caller must free result with talu_text_free.
pub export fn talu_get_chat_template_source(
    model_path: [*:0]const u8,
    out_source: *?[*:0]u8,
) callconv(.c) i32 {
    capi_error.clearError();
    out_source.* = null;

    const path = std.mem.span(model_path);
    const source = gen_config_mod.getChatTemplateSource(allocator, path) catch |err| {
        const code: error_codes.ErrorCode = switch (err) {
            error.FileNotFound, error.MissingChatTemplate => .template_not_found,
            else => error_codes.errorToCode(err),
        };
        capi_error.setErrorWithCode(code, "TemplateNotFound: no chat template at '{s}'", .{path});
        return @intFromEnum(code);
    };
    defer allocator.free(source);

    return copyToCString(source, out_source);
}

// =============================================================================
// Debug Mode: Span Tracking
// =============================================================================

/// Re-export span types from template module.
pub const CSpanSourceType = template_engine.CSpanSourceType;
pub const COutputSpan = template_engine.COutputSpan;

/// Convert and set output spans. Returns 0 on success, error code on failure.
fn convertOutputSpans(spans: ?[]const template_engine.OutputSpan, out_spans: *?[*]COutputSpan, out_count: *u32) i32 {
    const s = spans orelse {
        out_count.* = 0;
        return 0;
    };
    defer allocator.free(s);

    const span_list = template_engine.COutputSpanList.fromSpans(allocator, s) catch {
        capi_error.setErrorWithCode(.out_of_memory, "Span conversion failed", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    out_spans.* = span_list.spans.ptr;
    out_count.* = @intCast(span_list.spans.len);
    return 0;
}

/// Render with span tracking for debug visualization.
/// Caller must free out_rendered with talu_text_free and out_spans with talu_free_spans.
pub export fn talu_template_render_debug(
    template_str: [*:0]const u8,
    json_vars: [*:0]const u8,
    strict: bool,
    out_rendered: *?[*:0]u8,
    out_spans: *?[*]COutputSpan,
    out_span_count: *u32,
) callconv(.c) i32 {
    capi_error.clearError();
    out_rendered.* = null;
    out_spans.* = null;
    out_span_count.* = 0;

    const result = template_engine.renderFromJsonDebug(allocator, std.mem.span(template_str), std.mem.span(json_vars), strict);

    if (result.err) |err| {
        if (result.output) |o| allocator.free(o);
        if (result.spans) |s| allocator.free(s);
        capi_error.setError(err, "Template render failed", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    }

    const code = copyToCString(result.output.?, out_rendered);
    allocator.free(result.output.?);
    if (code != 0) {
        if (result.spans) |s| allocator.free(s);
        return code;
    }

    return convertOutputSpans(result.spans, out_spans, out_span_count);
}

/// Free spans allocated by talu_template_render_debug.
pub export fn talu_free_spans(spans: ?[*]COutputSpan, count: u32) callconv(.c) void {
    const s = spans orelse return;
    var span_list = template_engine.COutputSpanList{ .spans = s[0..count] };
    span_list.deinit(allocator);
}

// =============================================================================
// Validation
// =============================================================================

/// Validate template inputs without rendering.
/// Returns JSON: {"valid": bool, "required": [...], "optional": [...], "extra": [...]}
/// Caller must free result with talu_text_free.
pub export fn talu_template_validate(
    template_str: [*:0]const u8,
    json_vars: [*:0]const u8,
    out_result_json: *?[*:0]u8,
) callconv(.c) i32 {
    capi_error.clearError();
    out_result_json.* = null;

    var result = template_engine.validateJson(
        allocator,
        std.mem.span(template_str),
        std.mem.span(json_vars),
    ) catch |err| {
        const msg = template_engine.jsonErrorMessage(err);
        const code: error_codes.ErrorCode = if (err == error.OutOfMemory) .out_of_memory else .template_invalid_json;
        capi_error.setErrorWithCode(code, "TemplateValidation: {s}", .{msg});
        return @intFromEnum(code);
    };
    defer result.deinit(allocator);

    const json = template_engine.validationResultToJson(allocator, result) catch {
        capi_error.setErrorWithCode(.out_of_memory, "OutOfMemory: validation result", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    defer allocator.free(json);

    return copyToCString(json, out_result_json);
}

// =============================================================================
// Internal Helpers
// =============================================================================

fn copyToCString(src: []const u8, out: *?[*:0]u8) i32 {
    const cstr = allocator.allocSentinel(u8, src.len, 0) catch {
        capi_error.setErrorWithCode(.out_of_memory, "Allocation failed", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    @memcpy(cstr, src);
    out.* = cstr.ptr;
    return 0;
}

// =============================================================================
// Fuzz Tests
// =============================================================================

test "fuzz talu_template_render with random templates" {
    // Fuzz the template parser with arbitrary byte sequences.
    // The function should never crash, only return errors gracefully.
    try std.testing.fuzz({}, struct {
        fn testOne(_: void, input: []const u8) !void {
            // Need null-terminated strings for C API
            const template_z = std.testing.allocator.allocSentinel(u8, input.len, 0) catch return;
            defer std.testing.allocator.free(template_z[0 .. input.len + 1]);
            @memcpy(template_z[0..input.len], input);

            var out_rendered: ?[*:0]u8 = null;
            const result = talu_template_render(template_z.ptr, "{}", false, &out_rendered);

            // Clean up if successful
            if (result == 0) {
                if (out_rendered) |ptr| {
                    const slice = std.mem.span(ptr);
                    allocator.free(slice[0 .. slice.len + 1]);
                }
            } else {
                try std.testing.expect(out_rendered == null);
            }
            // Any return code is fine - we just want no crashes
        }
    }.testOne, .{});
}

test "fuzz talu_template_render" {
    // Alias fuzz target to match audit expectations.
    try std.testing.fuzz({}, struct {
        fn testOne(_: void, input: []const u8) !void {
            const alloc = std.testing.allocator;
            const template_z = alloc.allocSentinel(u8, input.len, 0) catch return;
            defer alloc.free(template_z[0 .. input.len + 1]);
            @memcpy(template_z[0..input.len], input);

            var out_rendered: ?[*:0]u8 = null;
            const result = talu_template_render(template_z.ptr, "{}", false, &out_rendered);
            if (result == 0) {
                if (out_rendered) |ptr| {
                    const slice = std.mem.span(ptr);
                    allocator.free(slice[0 .. slice.len + 1]);
                }
            } else {
                try std.testing.expect(out_rendered == null);
            }
        }
    }.testOne, .{});
}

test "fuzz talu_template_render with random JSON" {
    // Fuzz the JSON parser with arbitrary byte sequences.
    try std.testing.fuzz({}, struct {
        fn testOne(_: void, input: []const u8) !void {
            const json_z = std.testing.allocator.allocSentinel(u8, input.len, 0) catch return;
            defer std.testing.allocator.free(json_z[0 .. input.len + 1]);
            @memcpy(json_z[0..input.len], input);

            var out_rendered: ?[*:0]u8 = null;
            const result = talu_template_render("Hello {{ name }}", json_z.ptr, false, &out_rendered);

            if (result == 0) {
                if (out_rendered) |ptr| {
                    const slice = std.mem.span(ptr);
                    allocator.free(slice[0 .. slice.len + 1]);
                }
            } else {
                try std.testing.expect(out_rendered == null);
            }
        }
    }.testOne, .{});
}

test "fuzz talu_template_validate with random inputs" {
    // Fuzz validation with arbitrary template and JSON.
    try std.testing.fuzz({}, struct {
        fn testOne(_: void, input: []const u8) !void {
            // Split input in half for template and JSON
            const mid = input.len / 2;
            const template_part = input[0..mid];
            const json_part = input[mid..];

            const template_z = std.testing.allocator.allocSentinel(u8, template_part.len, 0) catch return;
            defer std.testing.allocator.free(template_z[0 .. template_part.len + 1]);
            @memcpy(template_z[0..template_part.len], template_part);

            const json_z = std.testing.allocator.allocSentinel(u8, json_part.len, 0) catch return;
            defer std.testing.allocator.free(json_z[0 .. json_part.len + 1]);
            @memcpy(json_z[0..json_part.len], json_part);

            var out_result: ?[*:0]u8 = null;
            const result = talu_template_validate(template_z.ptr, json_z.ptr, &out_result);

            if (result == 0) {
                if (out_result) |ptr| {
                    const slice = std.mem.span(ptr);
                    allocator.free(slice[0 .. slice.len + 1]);
                }
            } else {
                try std.testing.expect(out_result == null);
            }
        }
    }.testOne, .{});
}

test "fuzz talu_template_render_debug with random JSON" {
    try std.testing.fuzz({}, struct {
        fn testOne(_: void, input: []const u8) !void {
            const alloc = std.testing.allocator;
            const json_z = alloc.allocSentinel(u8, input.len, 0) catch return;
            defer alloc.free(json_z[0 .. input.len + 1]);
            @memcpy(json_z[0..input.len], input);

            var out_rendered: ?[*:0]u8 = null;
            var out_spans: ?[*]COutputSpan = null;
            var out_count: u32 = 0;
            const result = talu_template_render_debug("Hello {{ name }}", json_z.ptr, false, &out_rendered, &out_spans, &out_count);

            if (result == 0) {
                if (out_rendered) |ptr| {
                    const slice = std.mem.span(ptr);
                    allocator.free(slice[0 .. slice.len + 1]);
                }
                if (out_spans) |spans| {
                    talu_free_spans(spans, out_count);
                }
            } else {
                try std.testing.expect(out_rendered == null);
                try std.testing.expect(out_spans == null);
                try std.testing.expectEqual(@as(u32, 0), out_count);
            }
        }
    }.testOne, .{});
}
