//! C API for Provider Registry
//!
//! Exposes the provider registry to bindings (Rust, Python, etc.)
//! so they don't need to duplicate the provider list.

const std = @import("std");
const provider_mod = @import("../router/provider.zig");
const error_mod = @import("error.zig");
const error_codes = @import("error_codes.zig");
const setError = error_mod.setError;
const clearError = error_mod.clearError;

/// Provider info struct for C API.
/// Thread safety: Immutable after initialization.
pub const CProviderInfo = extern struct {
    /// Provider name (e.g., "vllm", "openai"). Null-terminated.
    name: [*:0]const u8,
    /// Default endpoint URL. Null-terminated.
    default_endpoint: [*:0]const u8,
    /// Environment variable name for API key, or null if not required.
    api_key_env: ?[*:0]const u8,
};

/// Get the number of registered providers.
pub export fn talu_provider_count() callconv(.c) usize {
    clearError();
    return provider_mod.PROVIDERS.len;
}

/// Get provider info by index.
/// Returns 0 on success, non-zero on error.
pub export fn talu_provider_get(index: usize, out: *CProviderInfo) callconv(.c) c_int {
    clearError();
    if (index >= provider_mod.PROVIDERS.len) {
        setError(error.InvalidArgument, "Provider index out of range: {}", .{index});
        return @intFromEnum(error_codes.errorToCode(error.InvalidArgument));
    }

    const p = provider_mod.PROVIDERS[index];
    out.* = std.mem.zeroes(CProviderInfo);
    out.name = @ptrCast(p.name.ptr);
    out.default_endpoint = @ptrCast(p.default_endpoint.ptr);
    out.api_key_env = if (p.api_key_env) |env| @ptrCast(env.ptr) else null;
    return 0;
}

/// Get provider info by name.
/// Returns 0 on success, non-zero if provider not found.
pub export fn talu_provider_get_by_name(name: [*:0]const u8, out: *CProviderInfo) callconv(.c) c_int {
    clearError();
    const name_slice = std.mem.sliceTo(name, 0);
    if (provider_mod.getByName(name_slice)) |p| {
        out.* = std.mem.zeroes(CProviderInfo);
        out.name = @ptrCast(p.name.ptr);
        out.default_endpoint = @ptrCast(p.default_endpoint.ptr);
        out.api_key_env = if (p.api_key_env) |env| @ptrCast(env.ptr) else null;
        return 0;
    }
    setError(error.InvalidArgument, "Unknown provider: {s}", .{name_slice});
    return @intFromEnum(error_codes.errorToCode(error.InvalidArgument));
}

/// Parse a provider-prefixed model ID (e.g., "vllm::org/model-name").
/// On success, writes provider info and sets model_id_out to the model ID portion.
/// Returns 0 on success, non-zero if no valid provider prefix.
pub export fn talu_provider_parse(
    model_id: [*:0]const u8,
    provider_out: *CProviderInfo,
    model_id_start: *usize,
    model_id_len: *usize,
) callconv(.c) c_int {
    clearError();
    const model_slice = std.mem.sliceTo(model_id, 0);
    if (provider_mod.parse(model_slice)) |result| {
        provider_out.* = std.mem.zeroes(CProviderInfo);
        provider_out.name = @ptrCast(result.provider.name.ptr);
        provider_out.default_endpoint = @ptrCast(result.provider.default_endpoint.ptr);
        provider_out.api_key_env = if (result.provider.api_key_env) |env| @ptrCast(env.ptr) else null;
        // Calculate offset into original string
        const separator_pos = std.mem.indexOf(u8, model_slice, "::").?;
        model_id_start.* = separator_pos + 2;
        model_id_len.* = result.model_id.len;
        return 0;
    }
    setError(error.InvalidArgument, "No valid provider prefix in model ID: {s}", .{model_slice});
    return @intFromEnum(error_codes.errorToCode(error.InvalidArgument));
}

/// Check if a model ID has a known provider prefix.
/// Returns 1 if true, 0 if false.
pub export fn talu_provider_has_prefix(model_id: [*:0]const u8) callconv(.c) c_int {
    clearError();
    const model_slice = std.mem.sliceTo(model_id, 0);
    return if (provider_mod.hasProviderPrefix(model_slice)) 1 else 0;
}

// =============================================================================
// Tests
// =============================================================================

test "talu_provider_count returns correct count" {
    try std.testing.expectEqual(@as(usize, 7), talu_provider_count());
}

test "talu_provider_get: valid index" {
    var info: CProviderInfo = undefined;
    try std.testing.expectEqual(@as(c_int, 0), talu_provider_get(0, &info));
    try std.testing.expectEqualStrings("vllm", std.mem.sliceTo(info.name, 0));
}

test "talu_provider_get: invalid index" {
    var info: CProviderInfo = undefined;
    try std.testing.expect(talu_provider_get(100, &info) != 0);
}

test "talu_provider_get_by_name: known provider" {
    var info: CProviderInfo = undefined;
    try std.testing.expectEqual(@as(c_int, 0), talu_provider_get_by_name("openai", &info));
    try std.testing.expectEqualStrings("openai", std.mem.sliceTo(info.name, 0));
    try std.testing.expectEqualStrings("https://api.openai.com/v1", std.mem.sliceTo(info.default_endpoint, 0));
}

test "talu_provider_get_by_name: unknown provider" {
    var info: CProviderInfo = undefined;
    try std.testing.expect(talu_provider_get_by_name("unknown", &info) != 0);
}

test "talu_provider_parse: valid provider prefix" {
    var info: CProviderInfo = undefined;
    var start: usize = undefined;
    var len: usize = undefined;
    try std.testing.expectEqual(@as(c_int, 0), talu_provider_parse("vllm::org/model-name", &info, &start, &len));
    try std.testing.expectEqualStrings("vllm", std.mem.sliceTo(info.name, 0));
    try std.testing.expectEqual(@as(usize, 6), start); // "vllm::" is 6 chars
    try std.testing.expectEqual(@as(usize, 14), len); // "org/model-name" is 14 chars
}

test "talu_provider_parse: no provider prefix" {
    var info: CProviderInfo = undefined;
    var start: usize = undefined;
    var len: usize = undefined;
    try std.testing.expect(talu_provider_parse("org/model-name", &info, &start, &len) != 0);
}

test "talu_provider_has_prefix" {
    try std.testing.expectEqual(@as(c_int, 1), talu_provider_has_prefix("vllm::model"));
    try std.testing.expectEqual(@as(c_int, 1), talu_provider_has_prefix("openai::gpt-4o"));
    try std.testing.expectEqual(@as(c_int, 0), talu_provider_has_prefix("org/model-name"));
    try std.testing.expectEqual(@as(c_int, 0), talu_provider_has_prefix("native::model"));
}
