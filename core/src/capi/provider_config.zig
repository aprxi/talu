//! C API for Provider Configuration (stateless, path-based).
//!
//! Exposes provider configuration management to bindings (Rust, Python, etc.).
//! Every function takes a `db_root` path parameter — no global state, no
//! init/deinit lifecycle.
//!
//! Thin glue: all logic lives in `router/provider_config.zig`.

const std = @import("std");
const router_mod = @import("../router/root.zig");
const provider_config_mod = router_mod.provider_config;
const http_engine_mod = router_mod.http_engine;
const capi_bridge = router_mod.capi_bridge;
const error_mod = @import("error.zig");
const error_codes = @import("error_codes.zig");

const clearError = error_mod.clearError;
const setError = error_mod.setError;
const setErrorWithCode = error_mod.setErrorWithCode;
const ErrorCode = error_codes.ErrorCode;
const errorToCode = error_codes.errorToCode;

const allocator = std.heap.c_allocator;

// =============================================================================
// C-ABI Types
// =============================================================================

/// Provider with merged runtime configuration.
/// Thread safety: Immutable after creation.
pub const CProviderWithConfig = extern struct {
    /// Provider name (e.g., "openai"). Null-terminated, owned.
    name: ?[*:0]const u8,
    /// Default endpoint URL. Null-terminated, owned.
    default_endpoint: ?[*:0]const u8,
    /// Env var name for API key, or null. Null-terminated, owned.
    api_key_env: ?[*:0]const u8,
    /// Whether the provider is enabled.
    enabled: u8,
    /// Whether an API key is available (from config or env).
    has_api_key: u8,
    /// Base URL override from config, or null. Null-terminated, owned.
    base_url_override: ?[*:0]const u8,
    /// Resolved effective endpoint. Null-terminated, owned.
    effective_endpoint: ?[*:0]const u8,
};

/// List of providers with config.
pub const CProviderConfigList = extern struct {
    items: ?[*]CProviderWithConfig,
    count: usize,
    error_code: i32,
};

/// Resolved credentials for a single provider (endpoint + API key).
pub const CProviderCredentials = extern struct {
    /// Resolved effective endpoint. Null-terminated, owned.
    effective_endpoint: ?[*:0]const u8,
    /// Resolved API key (from config or env), or null if unavailable. Owned.
    api_key: ?[*:0]const u8,
    /// Error code (0 = success).
    error_code: i32,
};

// =============================================================================
// Config Operations (stateless — every call takes db_root)
// =============================================================================

/// List all providers with their merged runtime configuration.
/// `db_root` is the KV store root path (e.g., "~/.talu/db/default/kv").
/// Caller must free the result with `talu_provider_config_list_free`.
pub export fn talu_provider_config_list(db_root: ?[*:0]const u8) callconv(.c) CProviderConfigList {
    clearError();

    const root_slice = if (db_root) |r| std.mem.sliceTo(r, 0) else {
        setErrorWithCode(.invalid_argument, "db_root is null", .{});
        var ret = std.mem.zeroes(CProviderConfigList);
        ret.error_code = @intFromEnum(ErrorCode.invalid_argument);
        return ret;
    };

    const items = provider_config_mod.listProvidersWithConfig(allocator, root_slice) catch |err| {
        setError(err, "Failed to list provider configs: {s}", .{@errorName(err)});
        var ret = std.mem.zeroes(CProviderConfigList);
        ret.error_code = @intFromEnum(errorToCode(err));
        return ret;
    };
    defer provider_config_mod.freeProvidersWithConfig(allocator, items);

    return convertProviderList(items);
}

/// Free result from `talu_provider_config_list`.
pub export fn talu_provider_config_list_free(list: ?*CProviderConfigList) callconv(.c) void {
    const ptr = list orelse return;
    if (ptr.items) |items_ptr| {
        const slice = items_ptr[0..ptr.count];
        for (slice) |*item| {
            if (item.name) |n| freeStr(n);
            if (item.default_endpoint) |d| freeStr(d);
            if (item.api_key_env) |a| freeStr(a);
            if (item.base_url_override) |b| freeStr(b);
            if (item.effective_endpoint) |e| freeStr(e);
        }
        allocator.free(slice);
    }
    ptr.items = null;
    ptr.count = 0;
}

/// Update configuration for a named provider using merge semantics.
///
/// `db_root`: KV store root path.
/// `name`: Provider name (e.g., "openai").
/// `enabled`: Trinary: -1=keep existing, 0=disable, 1=enable.
/// `api_key`: null=keep existing, ""=clear, non-empty=set.
/// `base_url`: null=keep existing, ""=clear, non-empty=set.
///
/// Returns 0 on success.
pub export fn talu_provider_config_set(
    db_root: ?[*:0]const u8,
    name: ?[*:0]const u8,
    enabled: i8,
    api_key: ?[*:0]const u8,
    base_url: ?[*:0]const u8,
) callconv(.c) c_int {
    clearError();

    const root_slice = if (db_root) |r| std.mem.sliceTo(r, 0) else {
        setErrorWithCode(.invalid_argument, "db_root is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    };

    const name_slice = if (name) |n| std.mem.sliceTo(n, 0) else {
        setErrorWithCode(.invalid_argument, "name is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    };

    // Build ConfigPatch with merge semantics.
    // enabled: -1=keep, 0=false, >=1=true.
    const patch = provider_config_mod.ConfigPatch{
        .enabled = if (enabled >= 0) (enabled != 0) else null,
        .api_key = mapFieldAction(api_key),
        .base_url = mapFieldAction(base_url),
    };

    provider_config_mod.updateProviderConfig(allocator, root_slice, name_slice, patch) catch |err| {
        setError(err, "Failed to set provider config for '{s}': {s}", .{ name_slice, @errorName(err) });
        return @intFromEnum(errorToCode(err));
    };

    return 0;
}

/// List models from all enabled remote providers.
/// Each model ID is prefixed with `{provider}::` (e.g., `openai::gpt-4o`).
/// Caller must free with `talu_provider_config_list_remote_models_free`.
pub export fn talu_provider_config_list_remote_models(db_root: ?[*:0]const u8) callconv(.c) capi_bridge.CRemoteModelListResult {
    clearError();

    const root_slice = if (db_root) |r| std.mem.sliceTo(r, 0) else {
        setErrorWithCode(.invalid_argument, "db_root is null", .{});
        var ret = std.mem.zeroes(capi_bridge.CRemoteModelListResult);
        ret.error_code = @intFromEnum(ErrorCode.invalid_argument);
        return ret;
    };

    const models = provider_config_mod.listRemoteModels(allocator, root_slice) catch |err| {
        setError(err, "Failed to list remote models: {s}", .{@errorName(err)});
        var ret = std.mem.zeroes(capi_bridge.CRemoteModelListResult);
        ret.error_code = @intFromEnum(errorToCode(err));
        return ret;
    };
    defer provider_config_mod.freeRemoteModels(allocator, models);

    return convertRemoteModels(models);
}

/// Free result from `talu_provider_config_list_remote_models`.
pub export fn talu_provider_config_list_remote_models_free(result: ?*capi_bridge.CRemoteModelListResult) callconv(.c) void {
    const ptr = result orelse return;
    capi_bridge.freeModelListResult(allocator, ptr);
}

/// Resolve the effective endpoint and API key for a named provider.
///
/// `db_root`: KV store root path.
/// `name`: Provider name (e.g., "openai").
///
/// Returns credentials with `error_code = 0` on success.
/// Caller must free with `talu_provider_config_resolve_credentials_free`.
pub export fn talu_provider_config_resolve_credentials(
    db_root: ?[*:0]const u8,
    name: ?[*:0]const u8,
) callconv(.c) CProviderCredentials {
    clearError();

    const root_slice = if (db_root) |r| std.mem.sliceTo(r, 0) else {
        setErrorWithCode(.invalid_argument, "db_root is null", .{});
        var ret = std.mem.zeroes(CProviderCredentials);
        ret.error_code = @intFromEnum(ErrorCode.invalid_argument);
        return ret;
    };

    const name_slice = if (name) |n| std.mem.sliceTo(n, 0) else {
        setErrorWithCode(.invalid_argument, "name is null", .{});
        var ret = std.mem.zeroes(CProviderCredentials);
        ret.error_code = @intFromEnum(ErrorCode.invalid_argument);
        return ret;
    };

    const provider_info = router_mod.provider.getByName(name_slice) orelse {
        setErrorWithCode(.invalid_argument, "Unknown provider: {s}", .{name_slice});
        var ret = std.mem.zeroes(CProviderCredentials);
        ret.error_code = @intFromEnum(ErrorCode.invalid_argument);
        return ret;
    };

    // Get config from KV (or defaults if unconfigured).
    const config = provider_config_mod.getProviderConfig(allocator, root_slice, name_slice) catch |err| {
        setError(err, "Failed to get provider config for '{s}': {s}", .{ name_slice, @errorName(err) });
        var ret = std.mem.zeroes(CProviderCredentials);
        ret.error_code = @intFromEnum(errorToCode(err));
        return ret;
    };
    const effective_config = config orelse provider_config_mod.ProviderConfig{};
    defer if (config) |c| c.deinit(allocator);

    // Resolve endpoint.
    const endpoint = provider_config_mod.resolveEndpointAlloc(allocator, provider_info, effective_config) catch |err| {
        setError(err, "Failed to resolve endpoint for '{s}': {s}", .{ name_slice, @errorName(err) });
        var ret = std.mem.zeroes(CProviderCredentials);
        ret.error_code = @intFromEnum(errorToCode(err));
        return ret;
    };
    defer allocator.free(endpoint);

    // Resolve API key (KV > env > null).
    const api_key = provider_config_mod.resolveApiKeyAlloc(allocator, root_slice, name_slice) catch null;
    defer if (api_key) |k| allocator.free(k);

    var ret = std.mem.zeroes(CProviderCredentials);
    ret.effective_endpoint = dupeSentinel(endpoint);
    ret.api_key = if (api_key) |k| dupeSentinel(k) else null;
    return ret;
}

/// Free result from `talu_provider_config_resolve_credentials`.
pub export fn talu_provider_config_resolve_credentials_free(creds: ?*CProviderCredentials) callconv(.c) void {
    const ptr = creds orelse return;
    if (ptr.effective_endpoint) |e| freeStr(e);
    if (ptr.api_key) |k| freeStr(k);
    ptr.effective_endpoint = null;
    ptr.api_key = null;
}

/// Result of a provider health check.
pub const CProviderHealthResult = extern struct {
    /// 1 if the provider responded successfully, 0 otherwise.
    ok: u8,
    /// Number of models listed (only meaningful when ok=1).
    model_count: usize,
    /// Error message if ok=0, null otherwise. Null-terminated, owned.
    error_message: ?[*:0]const u8,
};

/// Check connectivity to a provider by hitting its /models endpoint.
///
/// `db_root`: KV store root path.
/// `name`: Provider name (e.g., "openai").
///
/// Returns health result. Caller must free with `talu_provider_config_health_free`.
pub export fn talu_provider_config_health(
    db_root: ?[*:0]const u8,
    name: ?[*:0]const u8,
) callconv(.c) CProviderHealthResult {
    clearError();

    const root_slice = if (db_root) |r| std.mem.sliceTo(r, 0) else {
        setErrorWithCode(.invalid_argument, "db_root is null", .{});
        return .{ .ok = 0, .model_count = 0, .error_message = dupeSentinel("db_root is null") };
    };

    const name_slice = if (name) |n| std.mem.sliceTo(n, 0) else {
        setErrorWithCode(.invalid_argument, "name is null", .{});
        return .{ .ok = 0, .model_count = 0, .error_message = dupeSentinel("name is null") };
    };

    const result = provider_config_mod.checkProviderHealth(allocator, root_slice, name_slice);
    defer result.deinit(allocator);

    if (result.ok) {
        return .{ .ok = 1, .model_count = result.model_count, .error_message = null };
    } else {
        return .{
            .ok = 0,
            .model_count = 0,
            .error_message = if (result.error_message) |m| dupeSentinel(m) else null,
        };
    }
}

/// Free result from `talu_provider_config_health`.
pub export fn talu_provider_config_health_free(result: ?*CProviderHealthResult) callconv(.c) void {
    const ptr = result orelse return;
    if (ptr.error_message) |m| freeStr(m);
    ptr.error_message = null;
}

/// List models from a single named provider.
///
/// `db_root`: KV store root path.
/// `name`: Provider name (e.g., "openai").
///
/// Returns model list. Caller must free with `talu_provider_config_list_provider_models_free`.
pub export fn talu_provider_config_list_provider_models(
    db_root: ?[*:0]const u8,
    name: ?[*:0]const u8,
) callconv(.c) capi_bridge.CRemoteModelListResult {
    clearError();

    const root_slice = if (db_root) |r| std.mem.sliceTo(r, 0) else {
        setErrorWithCode(.invalid_argument, "db_root is null", .{});
        var ret = std.mem.zeroes(capi_bridge.CRemoteModelListResult);
        ret.error_code = @intFromEnum(ErrorCode.invalid_argument);
        return ret;
    };

    const name_slice = if (name) |n| std.mem.sliceTo(n, 0) else {
        setErrorWithCode(.invalid_argument, "name is null", .{});
        var ret = std.mem.zeroes(capi_bridge.CRemoteModelListResult);
        ret.error_code = @intFromEnum(ErrorCode.invalid_argument);
        return ret;
    };

    const models = provider_config_mod.listProviderModels(allocator, root_slice, name_slice) catch |err| {
        setError(err, "Failed to list models for provider '{s}': {s}", .{ name_slice, @errorName(err) });
        var ret = std.mem.zeroes(capi_bridge.CRemoteModelListResult);
        ret.error_code = @intFromEnum(errorToCode(err));
        return ret;
    };
    defer provider_config_mod.freeRemoteModels(allocator, models);

    return convertRemoteModels(models);
}

/// Free result from `talu_provider_config_list_provider_models`.
pub export fn talu_provider_config_list_provider_models_free(result: ?*capi_bridge.CRemoteModelListResult) callconv(.c) void {
    const ptr = result orelse return;
    capi_bridge.freeModelListResult(allocator, ptr);
}

// =============================================================================
// Internal Converters
// =============================================================================

/// Map a nullable C string pointer to a FieldAction for merge semantics.
/// null → .keep (don't change), "" → .clear (remove), non-empty → .set.
fn mapFieldAction(ptr: ?[*:0]const u8) provider_config_mod.FieldAction {
    const sentinel = ptr orelse return .keep;
    const slice = std.mem.sliceTo(sentinel, 0);
    if (slice.len == 0) return .clear;
    return .{ .set = slice };
}

/// Convert internal ProviderWithConfig slice to C-ABI list.
fn convertProviderList(items: []provider_config_mod.ProviderWithConfig) CProviderConfigList {
    if (items.len == 0) return std.mem.zeroes(CProviderConfigList);

    const c_items = allocator.alloc(CProviderWithConfig, items.len) catch {
        var ret = std.mem.zeroes(CProviderConfigList);
        ret.error_code = @intFromEnum(ErrorCode.out_of_memory);
        return ret;
    };

    for (items, 0..) |item, i| {
        var c = std.mem.zeroes(CProviderWithConfig);
        c.name = dupeSentinel(item.info.name);
        c.default_endpoint = dupeSentinel(item.info.default_endpoint);
        c.api_key_env = if (item.info.api_key_env) |env| dupeSentinel(env) else null;
        c.enabled = if (item.config.enabled) 1 else 0;
        c.has_api_key = if (item.has_api_key) 1 else 0;
        c.base_url_override = if (item.config.base_url) |url| dupeSentinel(url) else null;
        c.effective_endpoint = dupeSentinel(item.effective_endpoint);
        c_items[i] = c;
    }

    var ret = std.mem.zeroes(CProviderConfigList);
    ret.items = c_items.ptr;
    ret.count = items.len;
    return ret;
}

/// Convert internal ModelInfo slice to C-ABI CRemoteModelListResult.
fn convertRemoteModels(models: []http_engine_mod.ModelInfo) capi_bridge.CRemoteModelListResult {
    if (models.len == 0) return std.mem.zeroes(capi_bridge.CRemoteModelListResult);

    const c_models = allocator.alloc(capi_bridge.CRemoteModelInfo, models.len) catch {
        var ret = std.mem.zeroes(capi_bridge.CRemoteModelListResult);
        ret.error_code = @intFromEnum(ErrorCode.out_of_memory);
        return ret;
    };

    for (models, 0..) |model, i| {
        var c = std.mem.zeroes(capi_bridge.CRemoteModelInfo);
        c.id = dupeSentinelOwned(model.id);
        c.object = dupeSentinelOwned(model.object);
        c.created = model.created orelse 0;
        c.owned_by = dupeSentinelOwned(model.owned_by);
        c_models[i] = c;
    }

    var ret = std.mem.zeroes(capi_bridge.CRemoteModelListResult);
    ret.models = c_models.ptr;
    ret.count = models.len;
    return ret;
}

/// Dupe a borrowed slice to a sentinel-terminated C string. Returns null on failure.
fn dupeSentinel(src: []const u8) ?[*:0]const u8 {
    const duped = allocator.allocSentinel(u8, src.len, 0) catch return null;
    @memcpy(duped, src);
    return duped.ptr;
}

/// Dupe an owned (already allocated) slice to sentinel-terminated. Returns null on failure.
fn dupeSentinelOwned(src: []const u8) ?[*:0]u8 {
    const duped = allocator.allocSentinel(u8, src.len, 0) catch return null;
    @memcpy(duped, src);
    return duped.ptr;
}

/// Free a sentinel-terminated string allocated by this module.
fn freeStr(ptr: [*:0]const u8) void {
    const slice = std.mem.sliceTo(ptr, 0);
    allocator.free(slice.ptr[0 .. slice.len + 1]);
}

// =============================================================================
// Tests
// =============================================================================

test "talu_provider_config_list returns all providers" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const root_z = try std.testing.allocator.dupeZ(u8, root);
    defer std.testing.allocator.free(root_z);

    var list = talu_provider_config_list(root_z);
    defer talu_provider_config_list_free(&list);

    try std.testing.expectEqual(@as(i32, 0), list.error_code);
    try std.testing.expectEqual(@as(usize, 7), list.count);

    const items = list.items.?[0..list.count];
    for (items) |item| {
        try std.testing.expectEqual(@as(u8, 0), item.enabled);
    }
}

test "talu_provider_config_list rejects null db_root" {
    const list = talu_provider_config_list(null);
    try std.testing.expect(list.error_code != 0);
}

test "talu_provider_config_set and list round-trip" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const root_z = try std.testing.allocator.dupeZ(u8, root);
    defer std.testing.allocator.free(root_z);

    // Enable openai with custom endpoint.
    try std.testing.expectEqual(@as(c_int, 0), talu_provider_config_set(
        root_z,
        "openai",
        1,
        "sk-test-key",
        "http://custom:9000/v1",
    ));

    var list = talu_provider_config_list(root_z);
    defer talu_provider_config_list_free(&list);

    const items = list.items.?[0..list.count];
    var found = false;
    for (items) |item| {
        if (item.name != null and std.mem.eql(u8, std.mem.sliceTo(item.name.?, 0), "openai")) {
            try std.testing.expectEqual(@as(u8, 1), item.enabled);
            try std.testing.expectEqual(@as(u8, 1), item.has_api_key);
            try std.testing.expectEqualStrings("http://custom:9000/v1", std.mem.sliceTo(item.effective_endpoint.?, 0));
            found = true;
            break;
        }
    }
    try std.testing.expect(found);
}

test "talu_provider_config_set merge preserves existing fields" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const root_z = try std.testing.allocator.dupeZ(u8, root);
    defer std.testing.allocator.free(root_z);

    // Set initial config.
    try std.testing.expectEqual(@as(c_int, 0), talu_provider_config_set(
        root_z,
        "openai",
        1,
        "sk-secret",
        "http://custom/v1",
    ));

    // Toggle enabled only (pass null for api_key and base_url = keep).
    try std.testing.expectEqual(@as(c_int, 0), talu_provider_config_set(
        root_z,
        "openai",
        0,
        null,
        null,
    ));

    var list = talu_provider_config_list(root_z);
    defer talu_provider_config_list_free(&list);

    const items = list.items.?[0..list.count];
    var found = false;
    for (items) |item| {
        if (item.name != null and std.mem.eql(u8, std.mem.sliceTo(item.name.?, 0), "openai")) {
            // Enabled should be toggled.
            try std.testing.expectEqual(@as(u8, 0), item.enabled);
            // API key should be preserved.
            try std.testing.expectEqual(@as(u8, 1), item.has_api_key);
            // Effective endpoint should be the custom one.
            try std.testing.expectEqualStrings("http://custom/v1", std.mem.sliceTo(item.effective_endpoint.?, 0));
            found = true;
            break;
        }
    }
    try std.testing.expect(found);
}

test "talu_provider_config_set enabled=-1 preserves existing enabled state" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const root_z = try std.testing.allocator.dupeZ(u8, root);
    defer std.testing.allocator.free(root_z);

    // Enable provider.
    try std.testing.expectEqual(@as(c_int, 0), talu_provider_config_set(
        root_z, "openai", 1, "sk-key", null,
    ));

    // Update api_key only, keep enabled unchanged (enabled=-1).
    try std.testing.expectEqual(@as(c_int, 0), talu_provider_config_set(
        root_z, "openai", -1, "sk-new-key", null,
    ));

    var list = talu_provider_config_list(root_z);
    defer talu_provider_config_list_free(&list);

    const items = list.items.?[0..list.count];
    var found = false;
    for (items) |item| {
        if (item.name != null and std.mem.eql(u8, std.mem.sliceTo(item.name.?, 0), "openai")) {
            // Enabled should still be true (preserved).
            try std.testing.expectEqual(@as(u8, 1), item.enabled);
            try std.testing.expectEqual(@as(u8, 1), item.has_api_key);
            found = true;
            break;
        }
    }
    try std.testing.expect(found);
}

test "talu_provider_config_set rejects unknown provider" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const root_z = try std.testing.allocator.dupeZ(u8, root);
    defer std.testing.allocator.free(root_z);

    try std.testing.expect(talu_provider_config_set(root_z, "nonexistent", 1, null, null) != 0);
}

test "talu_provider_config_set rejects null name" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const root_z = try std.testing.allocator.dupeZ(u8, root);
    defer std.testing.allocator.free(root_z);

    try std.testing.expect(talu_provider_config_set(root_z, null, 1, null, null) != 0);
}

test "talu_provider_config_set rejects null db_root" {
    try std.testing.expect(talu_provider_config_set(null, "openai", 1, null, null) != 0);
}

test "talu_provider_config_list_free handles null" {
    talu_provider_config_list_free(null);
}

test "talu_provider_config_list_remote_models_free handles null" {
    talu_provider_config_list_remote_models_free(null);
}

test "CProviderWithConfig struct layout" {
    var info = std.mem.zeroes(CProviderWithConfig);
    info.enabled = 1;
    try std.testing.expect(info.name == null);
    try std.testing.expectEqual(@as(u8, 1), info.enabled);
}

test "CProviderConfigList struct layout" {
    const list = std.mem.zeroes(CProviderConfigList);
    try std.testing.expect(list.items == null);
    try std.testing.expectEqual(@as(usize, 0), list.count);
    try std.testing.expectEqual(@as(i32, 0), list.error_code);
}

test "talu_provider_config_resolve_credentials returns endpoint for known provider" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const root_z = try std.testing.allocator.dupeZ(u8, root);
    defer std.testing.allocator.free(root_z);

    var creds = talu_provider_config_resolve_credentials(root_z, "openai");
    defer talu_provider_config_resolve_credentials_free(&creds);

    try std.testing.expectEqual(@as(i32, 0), creds.error_code);
    try std.testing.expect(creds.effective_endpoint != null);
    try std.testing.expectEqualStrings("https://api.openai.com/v1", std.mem.sliceTo(creds.effective_endpoint.?, 0));
}

test "talu_provider_config_resolve_credentials uses config override" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const root_z = try std.testing.allocator.dupeZ(u8, root);
    defer std.testing.allocator.free(root_z);

    // Configure custom endpoint + API key.
    try std.testing.expectEqual(@as(c_int, 0), talu_provider_config_set(
        root_z,
        "openai",
        1,
        "sk-from-config",
        "http://custom:9000/v1",
    ));

    var creds = talu_provider_config_resolve_credentials(root_z, "openai");
    defer talu_provider_config_resolve_credentials_free(&creds);

    try std.testing.expectEqual(@as(i32, 0), creds.error_code);
    try std.testing.expectEqualStrings("http://custom:9000/v1", std.mem.sliceTo(creds.effective_endpoint.?, 0));
    try std.testing.expect(creds.api_key != null);
    try std.testing.expectEqualStrings("sk-from-config", std.mem.sliceTo(creds.api_key.?, 0));
}

test "talu_provider_config_resolve_credentials rejects unknown provider" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const root_z = try std.testing.allocator.dupeZ(u8, root);
    defer std.testing.allocator.free(root_z);

    const creds = talu_provider_config_resolve_credentials(root_z, "nonexistent");
    try std.testing.expect(creds.error_code != 0);
}

test "talu_provider_config_resolve_credentials rejects null args" {
    const c1 = talu_provider_config_resolve_credentials(null, "openai");
    try std.testing.expect(c1.error_code != 0);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const root_z = try std.testing.allocator.dupeZ(u8, root);
    defer std.testing.allocator.free(root_z);

    const c2 = talu_provider_config_resolve_credentials(root_z, null);
    try std.testing.expect(c2.error_code != 0);
}

test "talu_provider_config_resolve_credentials_free handles null" {
    talu_provider_config_resolve_credentials_free(null);
}
