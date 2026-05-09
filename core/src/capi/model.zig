//! C API for model metadata/config retrieval.
//!
//! This module is a thin FFI bridge. All retrieval/parsing logic lives in
//! `io/model_config.zig`.

const std = @import("std");
const capi_error = @import("error.zig");
const model_config = @import("io_pkg").model_config;

const alloc = std.heap.c_allocator;

/// Fetch HuggingFace model config and return normalized JSON.
///
/// Caller must free the returned string with `talu_text_free`.
pub export fn talu_model_hf_config_json(
    model_id: ?[*:0]const u8,
    revision: ?[*:0]const u8,
    endpoint_url: ?[*:0]const u8,
    token: ?[*:0]const u8,
    force_refresh: bool,
    include_size: bool,
) callconv(.c) ?[*:0]u8 {
    capi_error.clearError();

    const model = std.mem.span(model_id orelse {
        capi_error.setErrorWithCode(.invalid_argument, "model_id is null", .{});
        return null;
    });
    if (model.len == 0) {
        capi_error.setErrorWithCode(.invalid_argument, "model_id is empty", .{});
        return null;
    }

    const revision_slice: ?[]const u8 = if (revision) |r| std.mem.span(r) else null;
    const endpoint_slice: ?[]const u8 = if (endpoint_url) |e| std.mem.span(e) else null;
    const token_slice: ?[]const u8 = if (token) |t| std.mem.span(t) else null;

    const payload = model_config.fetchHfModelConfigJson(alloc, model, .{
        .token = token_slice,
        .endpoint_url = endpoint_slice,
        .revision = revision_slice,
        .force = force_refresh,
        .include_size = include_size,
    }) catch |err| {
        capi_error.setError(err, "failed to fetch model config", .{});
        return null;
    };
    defer alloc.free(payload);

    const out = alloc.allocSentinel(u8, payload.len, 0) catch {
        capi_error.setError(error.OutOfMemory, "failed to allocate model config payload", .{});
        return null;
    };
    @memcpy(out, payload);
    return out.ptr;
}

test "talu_model_hf_config_json validates required model_id" {
    const result = talu_model_hf_config_json(null, null, null, null, false, true);
    try std.testing.expect(result == null);
}
