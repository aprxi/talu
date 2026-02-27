//! Provider Configuration — Stateless, path-based runtime configuration for inference providers.
//!
//! Manages per-provider settings (enabled, api_key, base_url) backed by the
//! core KV store (`db/kv`). Every function opens the KV store at the given
//! `db_root` path, performs the operation, and closes it — no module-level
//! mutable state.
//!
//! ## Merge Semantics
//!
//! `updateProviderConfig` uses read-modify-write: reads existing config from KV,
//! applies only the fields provided in the patch, and writes the merged result.
//! This prevents PATCH updates from erasing fields not included in the request.
//!
//! ## Concurrent Model Listing
//!
//! `listRemoteModels` spawns one thread per enabled provider (via `std.Thread.spawn`)
//! for parallel HTTP queries. Latency = max(single provider) instead of sum(all).
//!
//! ## Thread Safety
//!
//! No module-level mutable state. Each call is isolated via its own KVStore instance.
//! Concurrent callers with the same `db_root` are serialized by the KV store's
//! filesystem-level locking.

const std = @import("std");
const provider_mod = @import("provider.zig");
const http_engine = @import("http_engine.zig");
const kv_store = @import("../db/kv/store.zig");
const log = @import("../log.zig");

const Allocator = std.mem.Allocator;

// ============================================================================
// Types
// ============================================================================

/// Per-provider runtime configuration. Caller owns allocated slices.
pub const ProviderConfig = struct {
    enabled: bool = false,
    /// API key (heap-allocated, owned). Null = not configured.
    api_key: ?[]const u8 = null,
    /// Base URL override (heap-allocated, owned). Null = use default.
    base_url: ?[]const u8 = null,

    /// Free owned allocations.
    pub fn deinit(self: *const ProviderConfig, allocator: Allocator) void {
        if (self.api_key) |k| allocator.free(k);
        if (self.base_url) |u| allocator.free(u);
    }

    /// Deep clone (caller owns returned config).
    pub fn clone(self: *const ProviderConfig, allocator: Allocator) !ProviderConfig {
        return ProviderConfig{
            .enabled = self.enabled,
            .api_key = if (self.api_key) |k| try allocator.dupe(u8, k) else null,
            .base_url = if (self.base_url) |u| try allocator.dupe(u8, u) else null,
        };
    }
};

/// Provider info merged with runtime configuration.
pub const ProviderWithConfig = struct {
    /// Static provider registry info (name, default_endpoint, api_key_env).
    info: provider_mod.Provider,
    /// Runtime configuration from KV store.
    config: ProviderConfig,
    /// True if an API key is available (from config or env var).
    has_api_key: bool,
    /// Resolved endpoint: config.base_url > env > info.default_endpoint.
    /// Heap-allocated, owned by caller.
    effective_endpoint: []const u8,

    pub fn deinit(self: *const ProviderWithConfig, allocator: Allocator) void {
        self.config.deinit(allocator);
        allocator.free(self.effective_endpoint);
    }
};

/// Describes a partial update to a provider's config.
/// Null fields mean "keep existing value". Used for merge semantics.
pub const ConfigPatch = struct {
    /// If non-null, set enabled to this value.
    enabled: ?bool = null,
    /// `.keep` = don't change, `.clear` = remove, `.set` = replace with value.
    api_key: FieldAction = .keep,
    /// `.keep` = don't change, `.clear` = remove, `.set` = replace with value.
    base_url: FieldAction = .keep,
};

/// Action for a nullable string field in a config patch.
pub const FieldAction = union(enum) {
    /// Don't change this field.
    keep,
    /// Remove this field (set to null).
    clear,
    /// Set this field to the given value.
    set: []const u8,
};

// KV key prefix for provider configs.
const kv_prefix = "provider:";

// ============================================================================
// Config Operations (stateless, path-based)
// ============================================================================

/// Update configuration for a named provider using merge semantics.
///
/// Opens the KV store at `db_root`, reads existing config, merges the patch,
/// and writes the result. This prevents partial updates from erasing fields.
///
/// Validates `name` against the static provider registry.
pub fn updateProviderConfig(allocator: Allocator, db_root: []const u8, name: []const u8, patch: ConfigPatch) !void {
    if (provider_mod.getByName(name) == null) {
        return error.UnknownProvider;
    }

    var store = try kv_store.KVStore.init(allocator, db_root, "provider_config");
    defer store.deinit();

    const kv_key = try std.fmt.allocPrint(allocator, "{s}{s}", .{ kv_prefix, name });
    defer allocator.free(kv_key);

    // Read existing config (or start from defaults).
    var existing = ProviderConfig{};
    if (try store.getCopy(allocator, kv_key)) |json_bytes| {
        defer allocator.free(json_bytes);
        existing = parseConfigJson(json_bytes) catch ProviderConfig{};
        // parseConfigJson returns borrowed slices into json_bytes.
        // Clone them so they survive after json_bytes is freed.
        const cloned = try existing.clone(allocator);
        existing = cloned;
    }
    defer existing.deinit(allocator);

    // Merge patch into existing config.
    var merged = ProviderConfig{
        .enabled = if (patch.enabled) |e| e else existing.enabled,
    };

    switch (patch.api_key) {
        .keep => merged.api_key = if (existing.api_key) |k| try allocator.dupe(u8, k) else null,
        .clear => merged.api_key = null,
        .set => |val| merged.api_key = try allocator.dupe(u8, val),
    }
    errdefer if (merged.api_key) |k| allocator.free(k);

    switch (patch.base_url) {
        .keep => merged.base_url = if (existing.base_url) |u| try allocator.dupe(u8, u) else null,
        .clear => merged.base_url = null,
        .set => |val| merged.base_url = try allocator.dupe(u8, val),
    }
    errdefer if (merged.base_url) |u| allocator.free(u);

    // Serialize and persist.
    const json = try serializeConfigJson(allocator, merged);
    defer allocator.free(json);

    try store.put(kv_key, json);

    // Free merged owned slices (they were duped for the store).
    merged.deinit(allocator);
}

/// Get the configuration for a named provider. Returns null if not configured.
/// Caller owns returned config and must call `deinit` on it.
pub fn getProviderConfig(allocator: Allocator, db_root: []const u8, name: []const u8) !?ProviderConfig {
    var store = try kv_store.KVStore.init(allocator, db_root, "provider_config");
    defer store.deinit();

    const kv_key = try std.fmt.allocPrint(allocator, "{s}{s}", .{ kv_prefix, name });
    defer allocator.free(kv_key);

    const json_bytes = try store.getCopy(allocator, kv_key) orelse return null;
    defer allocator.free(json_bytes);

    const parsed = parseConfigJson(json_bytes) catch return null;
    return try parsed.clone(allocator);
}

/// List all providers from the static registry merged with runtime configs.
/// Caller owns the returned slice and each element's allocated fields.
pub fn listProvidersWithConfig(allocator: Allocator, db_root: []const u8) ![]ProviderWithConfig {
    var store = try kv_store.KVStore.init(allocator, db_root, "provider_config");
    defer store.deinit();

    var result = std.ArrayListUnmanaged(ProviderWithConfig){};
    errdefer {
        for (result.items) |*item| item.deinit(allocator);
        result.deinit(allocator);
    }

    for (provider_mod.PROVIDERS) |p| {
        const kv_key = try std.fmt.allocPrint(allocator, "{s}{s}", .{ kv_prefix, p.name });
        defer allocator.free(kv_key);

        var config = ProviderConfig{};
        if (try store.getCopy(allocator, kv_key)) |json_bytes| {
            defer allocator.free(json_bytes);
            const parsed = parseConfigJson(json_bytes) catch ProviderConfig{};
            config = parsed.clone(allocator) catch ProviderConfig{};
        }
        errdefer config.deinit(allocator);

        const has_key = config.api_key != null or envKeyExists(allocator, p);
        const endpoint = try resolveEndpointAlloc(allocator, p, config);

        try result.append(allocator, .{
            .info = p,
            .config = config,
            .has_api_key = has_key,
            .effective_endpoint = endpoint,
        });
    }

    return result.items;
}

/// Free a slice returned by `listProvidersWithConfig`.
pub fn freeProvidersWithConfig(allocator: Allocator, items: []ProviderWithConfig) void {
    for (items) |*item| item.deinit(allocator);
    allocator.free(items);
}

/// Resolve the effective API key for a provider by name.
/// Priority: KV config > env var > null.
/// Returned slice is heap-allocated; caller owns it.
pub fn resolveApiKeyAlloc(allocator: Allocator, db_root: []const u8, name: []const u8) !?[]const u8 {
    const p = provider_mod.getByName(name) orelse return null;

    var store = try kv_store.KVStore.init(allocator, db_root, "provider_config");
    defer store.deinit();

    const kv_key = try std.fmt.allocPrint(allocator, "{s}{s}", .{ kv_prefix, name });
    defer allocator.free(kv_key);

    // 1. Config override from KV.
    if (try store.getCopy(allocator, kv_key)) |json_bytes| {
        defer allocator.free(json_bytes);
        if (parseConfigJson(json_bytes)) |cfg| {
            if (cfg.api_key) |key| {
                return try allocator.dupe(u8, key);
            }
        } else |_| {}
    }

    // 2. Environment variable fallback.
    if (p.api_key_env) |env_name| {
        if (std.process.getEnvVarOwned(allocator, env_name)) |val| {
            return val;
        } else |_| {}
    }

    return null;
}

// ============================================================================
// Health Check
// ============================================================================

/// Result of a provider health check.
pub const HealthResult = struct {
    /// True if the provider responded successfully.
    ok: bool,
    /// Number of models listed (only meaningful when ok=true).
    model_count: usize = 0,
    /// Error message if ok=false (heap-allocated, caller owns).
    error_message: ?[]const u8 = null,

    pub fn deinit(self: *const HealthResult, allocator: Allocator) void {
        if (self.error_message) |m| allocator.free(m);
    }
};

/// Check connectivity to a single provider by hitting its /models endpoint.
/// Resolves credentials from KV → env → defaults, then sends a GET request
/// with a 3-second timeout.
pub fn checkProviderHealth(allocator: Allocator, db_root: []const u8, name: []const u8) HealthResult {
    const p = provider_mod.getByName(name) orelse return .{
        .ok = false,
        .error_message = allocator.dupe(u8, "Unknown provider") catch null,
    };

    // Read config from KV.
    const cfg = blk: {
        var store = kv_store.KVStore.init(allocator, db_root, "provider_config") catch
            break :blk ProviderConfig{};
        defer store.deinit();

        const kv_key = std.fmt.allocPrint(allocator, "{s}{s}", .{ kv_prefix, name }) catch
            break :blk ProviderConfig{};
        defer allocator.free(kv_key);

        const json_bytes = (store.getCopy(allocator, kv_key) catch null) orelse
            break :blk ProviderConfig{};
        defer allocator.free(json_bytes);

        break :blk parseConfigJson(json_bytes) catch ProviderConfig{};
    };

    // Resolve endpoint + API key.
    const endpoint = resolveEndpointAlloc(allocator, p, cfg) catch return .{
        .ok = false,
        .error_message = allocator.dupe(u8, "Failed to resolve endpoint") catch null,
    };
    defer allocator.free(endpoint);

    const endpoint_z = allocator.dupeZ(u8, endpoint) catch return .{
        .ok = false,
        .error_message = allocator.dupe(u8, "Allocation failure") catch null,
    };
    defer allocator.free(endpoint_z);

    const api_key = resolveApiKeyInternal(allocator, p, cfg) catch null;
    const api_key_z: ?[:0]const u8 = if (api_key) |k| zblk: {
        const z = allocator.dupeZ(u8, k) catch {
            allocator.free(k);
            break :zblk null;
        };
        allocator.free(k);
        break :zblk z;
    } else null;
    defer if (api_key_z) |k| allocator.free(k);

    // Create engine and hit /models.
    var engine = http_engine.HttpEngine.init(allocator, .{
        .base_url = endpoint_z,
        .api_key = api_key_z,
        .model = "unused",
        .timeout_ms = 3_000,
        .connect_timeout_ms = 3_000,
        .max_retries = 0,
    }) catch return .{
        .ok = false,
        .error_message = allocator.dupe(u8, "Failed to initialize HTTP engine") catch null,
    };
    defer engine.deinit();

    const result = engine.listModels() catch |err| {
        return .{
            .ok = false,
            .error_message = std.fmt.allocPrint(allocator, "{s}", .{@errorName(err)}) catch null,
        };
    };
    defer result.deinit(allocator);

    return .{
        .ok = true,
        .model_count = result.models.len,
    };
}

// ============================================================================
// Model Aggregation (concurrent)
// ============================================================================

/// Context for a per-provider model-fetching thread.
const FetchThreadCtx = struct {
    allocator: Allocator,
    name: []const u8,
    endpoint_z: [:0]const u8,
    api_key_z: ?[:0]const u8,
    /// Output models (written by thread, read by caller after join).
    models: ?[]http_engine.ModelInfo = null,
    /// True if the fetch failed (models remains null).
    failed: bool = false,
};

/// Thread entry point: queries a single provider's /models endpoint.
fn fetchModelsThread(ctx: *FetchThreadCtx) void {
    var engine = http_engine.HttpEngine.init(ctx.allocator, .{
        .base_url = ctx.endpoint_z,
        .api_key = ctx.api_key_z,
        .model = "unused",
        .timeout_ms = 3_000,
        .connect_timeout_ms = 3_000,
        .max_retries = 0,
    }) catch {
        ctx.failed = true;
        return;
    };
    defer engine.deinit();

    const result = engine.listModels() catch {
        ctx.failed = true;
        return;
    };

    ctx.models = result.models;
}

/// Query all enabled remote providers concurrently and return a merged list.
/// Each model ID is prefixed with `{provider}::` (e.g., `openai::gpt-4o`).
/// Providers that fail to respond are silently skipped (warning logged).
/// Caller owns the returned slice and each ModelInfo.
pub fn listRemoteModels(allocator: Allocator, db_root: []const u8) ![]http_engine.ModelInfo {
    // Collect enabled providers' connection info.
    const max_providers = provider_mod.PROVIDERS.len;
    var contexts: [max_providers]FetchThreadCtx = undefined;
    var ctx_count: usize = 0;

    // Scope the KV store so the file lock is released before spawning HTTP
    // threads. This prevents config-save requests from blocking during the
    // potentially multi-second network I/O.
    {
        var store = try kv_store.KVStore.init(allocator, db_root, "provider_config");
        defer store.deinit();

        for (provider_mod.PROVIDERS) |p| {
            const kv_key = std.fmt.allocPrint(allocator, "{s}{s}", .{ kv_prefix, p.name }) catch continue;
            defer allocator.free(kv_key);

            const json_bytes = (store.getCopy(allocator, kv_key) catch continue) orelse continue;
            defer allocator.free(json_bytes);

            const cfg = parseConfigJson(json_bytes) catch continue;
            if (!cfg.enabled) continue;

            const endpoint = resolveEndpointAlloc(allocator, p, cfg) catch continue;
            errdefer allocator.free(endpoint);
            const endpoint_z = allocator.dupeZ(u8, endpoint) catch continue;
            allocator.free(endpoint);

            const api_key = resolveApiKeyInternal(allocator, p, cfg) catch null;
            const api_key_z: ?[:0]const u8 = if (api_key) |k| blk: {
                const z = allocator.dupeZ(u8, k) catch {
                    allocator.free(k);
                    break :blk null;
                };
                allocator.free(k);
                break :blk z;
            } else null;

            contexts[ctx_count] = .{
                .allocator = allocator,
                .name = p.name,
                .endpoint_z = endpoint_z,
                .api_key_z = api_key_z,
            };
            ctx_count += 1;
        }
    } // KV file lock released here.

    // Spawn threads for concurrent fetching.
    var threads: [max_providers]?std.Thread = .{null} ** max_providers;
    for (0..ctx_count) |i| {
        threads[i] = std.Thread.spawn(.{}, fetchModelsThread, .{&contexts[i]}) catch null;
    }

    // Join all threads.
    for (0..ctx_count) |i| {
        if (threads[i]) |thread| thread.join();
    }

    // Merge results.
    var all_models = std.ArrayListUnmanaged(http_engine.ModelInfo){};
    errdefer {
        for (all_models.items) |*m| m.deinit(allocator);
        all_models.deinit(allocator);
    }

    for (0..ctx_count) |i| {
        const ctx = &contexts[i];
        defer {
            allocator.free(ctx.endpoint_z);
            if (ctx.api_key_z) |k| allocator.free(k);
        }

        if (ctx.failed or ctx.models == null) {
            if (ctx.failed) {
                log.warn("provider_config", "Failed to list models for provider", .{
                    .provider = ctx.name,
                });
            }
            continue;
        }

        const models = ctx.models.?;
        defer allocator.free(models);

        for (models) |model| {
            const prefixed_id = std.fmt.allocPrint(allocator, "{s}::{s}", .{ ctx.name, model.id }) catch continue;
            errdefer allocator.free(prefixed_id);

            const owned_object = allocator.dupe(u8, model.object) catch continue;
            errdefer allocator.free(owned_object);
            const owned_by = allocator.dupe(u8, model.owned_by) catch continue;

            allocator.free(model.id);

            all_models.append(allocator, .{
                .id = prefixed_id,
                .object = owned_object,
                .created = model.created,
                .owned_by = owned_by,
            }) catch continue;
        }
    }

    return all_models.toOwnedSlice(allocator);
}

/// Free models returned by `listRemoteModels`.
pub fn freeRemoteModels(allocator: Allocator, models: []http_engine.ModelInfo) void {
    for (models) |*m| m.deinit(allocator);
    allocator.free(models);
}

// ============================================================================
// Single-Provider Model Listing
// ============================================================================

/// List models from a single named provider.
/// Resolves credentials from KV → env → defaults, queries the provider's
/// /models endpoint, and returns the model list (without provider:: prefix).
/// Caller owns the returned slice and each ModelInfo.
pub fn listProviderModels(allocator: Allocator, db_root: []const u8, name: []const u8) ![]http_engine.ModelInfo {
    const p = provider_mod.getByName(name) orelse return error.UnknownProvider;

    // Read config from KV.
    const cfg = blk: {
        var store = kv_store.KVStore.init(allocator, db_root, "provider_config") catch
            break :blk ProviderConfig{};
        defer store.deinit();

        const kv_key = std.fmt.allocPrint(allocator, "{s}{s}", .{ kv_prefix, name }) catch
            break :blk ProviderConfig{};
        defer allocator.free(kv_key);

        const json_bytes = (store.getCopy(allocator, kv_key) catch null) orelse
            break :blk ProviderConfig{};
        defer allocator.free(json_bytes);

        break :blk parseConfigJson(json_bytes) catch ProviderConfig{};
    };

    // Resolve endpoint + API key.
    const endpoint = resolveEndpointAlloc(allocator, p, cfg) catch return error.ResolveFailed;
    defer allocator.free(endpoint);

    const endpoint_z = try allocator.dupeZ(u8, endpoint);
    defer allocator.free(endpoint_z);

    const api_key = resolveApiKeyInternal(allocator, p, cfg) catch null;
    const api_key_z: ?[:0]const u8 = if (api_key) |k| zblk: {
        const z = allocator.dupeZ(u8, k) catch {
            allocator.free(k);
            break :zblk null;
        };
        allocator.free(k);
        break :zblk z;
    } else null;
    defer if (api_key_z) |k| allocator.free(k);

    var engine = http_engine.HttpEngine.init(allocator, .{
        .base_url = endpoint_z,
        .api_key = api_key_z,
        .model = "unused",
        .timeout_ms = 5_000,
        .connect_timeout_ms = 3_000,
        .max_retries = 0,
    }) catch return error.EngineInitFailed;
    defer engine.deinit();

    const result = try engine.listModels();
    // Transfer ownership of models slice to caller (don't deinit result).
    // The caller will free via freeRemoteModels.
    return result.models;
}

// ============================================================================
// Endpoint / Key Resolution (pure helpers, no KV access)
// ============================================================================

/// Resolve the effective base URL for a provider.
/// Priority: config.base_url > env({NAME}_ENDPOINT) > info.default_endpoint.
/// Returned slice is heap-allocated; caller owns it.
pub fn resolveEndpointAlloc(allocator: Allocator, p: provider_mod.Provider, config: ProviderConfig) ![]const u8 {
    // 1. Config override.
    if (config.base_url) |url| {
        return try allocator.dupe(u8, url);
    }

    // 2. Environment variable: {PROVIDER}_ENDPOINT.
    const env_key = try std.fmt.allocPrint(allocator, "{s}_ENDPOINT", .{upperName(p.name)});
    defer allocator.free(env_key);
    const env_key_z = try allocator.dupeZ(u8, env_key);
    defer allocator.free(env_key_z);

    if (std.process.getEnvVarOwned(allocator, env_key_z)) |val| {
        return val;
    } else |_| {}

    // 3. Registry default.
    return try allocator.dupe(u8, p.default_endpoint);
}

// ============================================================================
// Internal Helpers
// ============================================================================

/// Resolve API key without KV access (for use when config is already loaded).
fn resolveApiKeyInternal(allocator: Allocator, p: provider_mod.Provider, cfg: ProviderConfig) !?[]const u8 {
    if (cfg.api_key) |key| {
        return try allocator.dupe(u8, key);
    }
    if (p.api_key_env) |env_name| {
        if (std.process.getEnvVarOwned(allocator, env_name)) |val| {
            return val;
        } else |_| {}
    }
    return null;
}

/// Check if an env-var API key exists for a provider (without allocating the value).
fn envKeyExists(allocator: Allocator, p: provider_mod.Provider) bool {
    const env_name = p.api_key_env orelse return false;
    if (std.process.getEnvVarOwned(allocator, env_name)) |val| {
        allocator.free(val);
        return true;
    } else |_| {
        return false;
    }
}

/// Convert a provider name to uppercase for env var lookup.
/// Returns a slice from a thread-local buffer — not heap-allocated.
fn upperName(name: []const u8) []const u8 {
    const S = struct {
        /// Thread-local buffer for uppercase conversion.
        /// threadlocal: pointers/values cannot be shared across threads.
        threadlocal var buf: [32]u8 = undefined;
    };
    const len = @min(name.len, S.buf.len);
    for (name[0..len], 0..len) |c, i| {
        S.buf[i] = std.ascii.toUpper(c);
    }
    return S.buf[0..len];
}

/// Serialize a ProviderConfig to JSON bytes.
fn serializeConfigJson(allocator: Allocator, config: ProviderConfig) ![]u8 {
    var buf = std.ArrayListUnmanaged(u8){};
    errdefer buf.deinit(allocator);
    var writer = buf.writer(allocator);

    try writer.writeAll("{\"enabled\":");
    try writer.writeAll(if (config.enabled) "true" else "false");

    if (config.api_key) |key| {
        try writer.writeAll(",\"api_key\":\"");
        try writeJsonEscaped(writer, key);
        try writer.writeByte('"');
    }
    if (config.base_url) |url| {
        try writer.writeAll(",\"base_url\":\"");
        try writeJsonEscaped(writer, url);
        try writer.writeByte('"');
    }
    try writer.writeByte('}');

    return buf.toOwnedSlice(allocator);
}

/// Parse a ProviderConfig from JSON bytes. Returned config does NOT own its
/// slices — they point into the input `json` buffer.
fn parseConfigJson(json: []const u8) !ProviderConfig {
    var config = ProviderConfig{};
    config.enabled = findJsonBool(json, "enabled") orelse false;
    config.api_key = findJsonString(json, "api_key");
    config.base_url = findJsonString(json, "base_url");
    return config;
}

/// Find a boolean value in a flat JSON object.
fn findJsonBool(json: []const u8, comptime key: []const u8) ?bool {
    const needle_true = std.fmt.comptimePrint("\"{s}\":true", .{key});
    const needle_false = std.fmt.comptimePrint("\"{s}\":false", .{key});
    if (std.mem.indexOf(u8, json, needle_true) != null) return true;
    if (std.mem.indexOf(u8, json, needle_false) != null) return false;
    return null;
}

/// Find a string value in a flat JSON object. Returns a slice into `json`.
fn findJsonString(json: []const u8, comptime key: []const u8) ?[]const u8 {
    const needle = std.fmt.comptimePrint("\"{s}\":\"", .{key});
    const start_idx = (std.mem.indexOf(u8, json, needle) orelse return null) + needle.len;
    const end_idx = std.mem.indexOfPos(u8, json, start_idx, "\"") orelse return null;
    const value = json[start_idx..end_idx];
    if (value.len == 0) return null;
    return value;
}

/// Write a JSON-escaped string (handles \, ", control chars).
fn writeJsonEscaped(writer: anytype, s: []const u8) !void {
    for (s) |c| {
        switch (c) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            else => {
                if (c < 0x20) {
                    try writer.print("\\u{x:0>4}", .{c});
                } else {
                    try writer.writeByte(c);
                }
            },
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "ProviderConfig.clone round-trips" {
    const allocator = std.testing.allocator;
    const original = ProviderConfig{
        .enabled = true,
        .api_key = "sk-test-key",
        .base_url = "http://custom:8000/v1",
    };
    const cloned = try original.clone(allocator);
    defer cloned.deinit(allocator);

    try std.testing.expect(cloned.enabled);
    try std.testing.expectEqualStrings("sk-test-key", cloned.api_key.?);
    try std.testing.expectEqualStrings("http://custom:8000/v1", cloned.base_url.?);
    try std.testing.expect(cloned.api_key.?.ptr != original.api_key.?.ptr);
}

test "ProviderConfig.clone handles nulls" {
    const allocator = std.testing.allocator;
    const original = ProviderConfig{ .enabled = false };
    const cloned = try original.clone(allocator);
    defer cloned.deinit(allocator);

    try std.testing.expect(!cloned.enabled);
    try std.testing.expect(cloned.api_key == null);
    try std.testing.expect(cloned.base_url == null);
}

test "serializeConfigJson produces valid JSON" {
    const allocator = std.testing.allocator;
    const config = ProviderConfig{
        .enabled = true,
        .api_key = "sk-123",
        .base_url = "http://example.com/v1",
    };
    const json = try serializeConfigJson(allocator, config);
    defer allocator.free(json);

    try std.testing.expectEqualStrings(
        "{\"enabled\":true,\"api_key\":\"sk-123\",\"base_url\":\"http://example.com/v1\"}",
        json,
    );
}

test "serializeConfigJson omits null fields" {
    const allocator = std.testing.allocator;
    const config = ProviderConfig{ .enabled = false };
    const json = try serializeConfigJson(allocator, config);
    defer allocator.free(json);

    try std.testing.expectEqualStrings("{\"enabled\":false}", json);
}

test "parseConfigJson round-trips with serializeConfigJson" {
    const allocator = std.testing.allocator;
    const original = ProviderConfig{
        .enabled = true,
        .api_key = "test-key",
        .base_url = "http://localhost:8000/v1",
    };
    const json = try serializeConfigJson(allocator, original);
    defer allocator.free(json);

    const parsed = try parseConfigJson(json);
    try std.testing.expect(parsed.enabled);
    try std.testing.expectEqualStrings("test-key", parsed.api_key.?);
    try std.testing.expectEqualStrings("http://localhost:8000/v1", parsed.base_url.?);
}

test "parseConfigJson handles missing fields" {
    const parsed = try parseConfigJson("{\"enabled\":false}");
    try std.testing.expect(!parsed.enabled);
    try std.testing.expect(parsed.api_key == null);
    try std.testing.expect(parsed.base_url == null);
}

test "findJsonBool detects true and false" {
    try std.testing.expectEqual(true, findJsonBool("{\"enabled\":true}", "enabled").?);
    try std.testing.expectEqual(false, findJsonBool("{\"enabled\":false}", "enabled").?);
    try std.testing.expect(findJsonBool("{}", "enabled") == null);
}

test "findJsonString extracts value" {
    const json = "{\"api_key\":\"sk-123\",\"base_url\":\"http://x\"}";
    try std.testing.expectEqualStrings("sk-123", findJsonString(json, "api_key").?);
    try std.testing.expectEqualStrings("http://x", findJsonString(json, "base_url").?);
    try std.testing.expect(findJsonString(json, "missing") == null);
}

test "upperName converts to uppercase" {
    try std.testing.expectEqualStrings("OPENAI", upperName("openai"));
    try std.testing.expectEqualStrings("VLLM", upperName("vllm"));
    try std.testing.expectEqualStrings("OPENROUTER", upperName("openrouter"));
}

test "writeJsonEscaped handles special characters" {
    const allocator = std.testing.allocator;
    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);

    try writeJsonEscaped(buf.writer(allocator), "hello \"world\"\nline2");
    try std.testing.expectEqualStrings("hello \\\"world\\\"\\nline2", buf.items);
}

test "updateProviderConfig persists to KV" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(root);

    try updateProviderConfig(allocator, root, "openai", .{
        .enabled = true,
        .api_key = .{ .set = "sk-test" },
        .base_url = .{ .set = "http://custom:9000/v1" },
    });

    // Verify by reading back.
    const cfg = (try getProviderConfig(allocator, root, "openai")).?;
    defer cfg.deinit(allocator);

    try std.testing.expect(cfg.enabled);
    try std.testing.expectEqualStrings("sk-test", cfg.api_key.?);
    try std.testing.expectEqualStrings("http://custom:9000/v1", cfg.base_url.?);
}

test "updateProviderConfig rejects unknown provider" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(root);

    try std.testing.expectError(error.UnknownProvider, updateProviderConfig(allocator, root, "nonexistent", .{
        .enabled = true,
    }));
}

test "updateProviderConfig merge preserves existing fields" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(root);

    // Set initial config with api_key and base_url.
    try updateProviderConfig(allocator, root, "openai", .{
        .enabled = true,
        .api_key = .{ .set = "sk-secret" },
        .base_url = .{ .set = "http://custom/v1" },
    });

    // Toggle only enabled, keeping api_key and base_url.
    try updateProviderConfig(allocator, root, "openai", .{
        .enabled = false,
        // api_key and base_url default to .keep
    });

    // Verify api_key and base_url are preserved.
    const cfg = (try getProviderConfig(allocator, root, "openai")).?;
    defer cfg.deinit(allocator);

    try std.testing.expect(!cfg.enabled);
    try std.testing.expectEqualStrings("sk-secret", cfg.api_key.?);
    try std.testing.expectEqualStrings("http://custom/v1", cfg.base_url.?);
}

test "updateProviderConfig merge can clear a field" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(root);

    // Set config.
    try updateProviderConfig(allocator, root, "vllm", .{
        .enabled = true,
        .api_key = .{ .set = "key-1" },
        .base_url = .{ .set = "http://gpu:8000/v1" },
    });

    // Clear api_key, keep base_url.
    try updateProviderConfig(allocator, root, "vllm", .{
        .api_key = .clear,
    });

    const cfg = (try getProviderConfig(allocator, root, "vllm")).?;
    defer cfg.deinit(allocator);

    try std.testing.expect(cfg.enabled);
    try std.testing.expect(cfg.api_key == null);
    try std.testing.expectEqualStrings("http://gpu:8000/v1", cfg.base_url.?);
}

test "getProviderConfig returns null for unconfigured provider" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(root);

    try std.testing.expect((try getProviderConfig(allocator, root, "openai")) == null);
}

test "listProvidersWithConfig returns all 7 providers" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(root);

    const items = try listProvidersWithConfig(allocator, root);
    defer freeProvidersWithConfig(allocator, items);

    try std.testing.expectEqual(@as(usize, 7), items.len);

    for (items) |item| {
        try std.testing.expect(!item.config.enabled);
    }
}

test "listProvidersWithConfig reflects updateProviderConfig changes" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(root);

    try updateProviderConfig(allocator, root, "openai", .{
        .enabled = true,
        .api_key = .{ .set = "sk-test" },
    });

    const items = try listProvidersWithConfig(allocator, root);
    defer freeProvidersWithConfig(allocator, items);

    var found = false;
    for (items) |item| {
        if (std.mem.eql(u8, item.info.name, "openai")) {
            try std.testing.expect(item.config.enabled);
            try std.testing.expect(item.has_api_key);
            found = true;
            break;
        }
    }
    try std.testing.expect(found);
}

test "resolveEndpointAlloc uses config override first" {
    const allocator = std.testing.allocator;
    const p = provider_mod.getByName("openai").?;
    const config = ProviderConfig{
        .base_url = "http://custom:8080/v1",
    };
    const endpoint = try resolveEndpointAlloc(allocator, p, config);
    defer allocator.free(endpoint);

    try std.testing.expectEqualStrings("http://custom:8080/v1", endpoint);
}

test "resolveEndpointAlloc falls back to default" {
    const allocator = std.testing.allocator;
    const p = provider_mod.getByName("openai").?;
    const config = ProviderConfig{};
    const endpoint = try resolveEndpointAlloc(allocator, p, config);
    defer allocator.free(endpoint);

    try std.testing.expectEqualStrings("https://api.openai.com/v1", endpoint);
}

test "KV persistence survives across calls" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(root);

    // First call: set config.
    try updateProviderConfig(allocator, root, "vllm", .{
        .enabled = true,
        .base_url = .{ .set = "http://gpu:8000/v1" },
    });

    // Second call: verify config loaded from KV.
    const cfg = (try getProviderConfig(allocator, root, "vllm")).?;
    defer cfg.deinit(allocator);

    try std.testing.expect(cfg.enabled);
    try std.testing.expectEqualStrings("http://gpu:8000/v1", cfg.base_url.?);
}
