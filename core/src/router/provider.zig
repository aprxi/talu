//! Provider Registry - Default endpoints for OpenAI-compatible providers.
//!
//! This module defines the canonical list of supported remote inference providers
//! with their default endpoints and API key environment variables.
//!
//! ## Supported Providers
//!
//! | Provider    | Default Endpoint                    | API Key Env       |
//! |-------------|-------------------------------------|-------------------|
//! | vllm        | http://localhost:8000/v1            | VLLM_API_KEY      |
//! | ollama      | http://localhost:11434/v1           | (none)            |
//! | llamacpp    | http://localhost:8080/v1            | (none)            |
//! | lmstudio    | http://localhost:1234/v1            | (none)            |
//! | localai     | http://localhost:8080/v1            | (none)            |
//! | openai      | https://api.openai.com/v1           | OPENAI_API_KEY    |
//! | openrouter  | https://openrouter.ai/api/v1        | OPENROUTER_API_KEY|
//!
//! ## Usage
//!
//! ```zig
//! const provider = @import("router/provider.zig");
//!
//! // Parse "vllm::org/model-name" -> provider config + model ID
//! if (provider.parse("vllm::org/model-name")) |result| {
//!     std.debug.print("Provider: {s}\n", .{result.provider.name});
//!     std.debug.print("Endpoint: {s}\n", .{result.provider.default_endpoint});
//!     std.debug.print("Model: {s}\n", .{result.model_id});
//! }
//!
//! // Get provider by name
//! if (provider.getByName("openai")) |p| {
//!     std.debug.print("OpenAI endpoint: {s}\n", .{p.default_endpoint});
//! }
//! ```

const std = @import("std");

/// Provider configuration.
pub const Provider = struct {
    /// Provider name (e.g., "vllm", "openai").
    name: []const u8,
    /// Default base URL for the provider's API.
    default_endpoint: []const u8,
    /// Environment variable name for API key (null if not required).
    api_key_env: ?[]const u8,
};

/// Result of parsing a provider-prefixed model ID.
pub const ParseResult = struct {
    /// The matched provider configuration.
    provider: Provider,
    /// The model ID after the provider prefix.
    model_id: []const u8,
};

/// Canonical list of supported providers.
/// All providers use OpenAI-compatible API format.
pub const PROVIDERS = [_]Provider{
    // Local servers
    .{
        .name = "vllm",
        .default_endpoint = "http://localhost:8000/v1",
        .api_key_env = "VLLM_API_KEY",
    },
    .{
        .name = "ollama",
        .default_endpoint = "http://localhost:11434/v1",
        .api_key_env = null,
    },
    .{
        .name = "llamacpp",
        .default_endpoint = "http://localhost:8080/v1",
        .api_key_env = null,
    },
    .{
        .name = "lmstudio",
        .default_endpoint = "http://localhost:1234/v1",
        .api_key_env = null,
    },
    .{
        .name = "localai",
        .default_endpoint = "http://localhost:8080/v1",
        .api_key_env = null,
    },
    // Cloud providers
    .{
        .name = "openai",
        .default_endpoint = "https://api.openai.com/v1",
        .api_key_env = "OPENAI_API_KEY",
    },
    .{
        .name = "openrouter",
        .default_endpoint = "https://openrouter.ai/api/v1",
        .api_key_env = "OPENROUTER_API_KEY",
    },
};

/// Get provider by name.
/// Returns null if the provider is not found.
pub fn getByName(name: []const u8) ?Provider {
    for (PROVIDERS) |p| {
        if (std.mem.eql(u8, p.name, name)) {
            return p;
        }
    }
    return null;
}

/// Parse a provider-prefixed model ID.
/// Returns null if no provider prefix is found or provider is unknown.
///
/// Examples:
///   "vllm::org/model-name" -> { provider: vllm, model_id: "org/model-name" }
///   "openai::gpt-4o" -> { provider: openai, model_id: "gpt-4o" }
///   "org/model-name" -> null (no provider prefix)
///   "native::model" -> null (native is not a remote provider)
///   "unknown::model" -> null (unknown provider)
pub fn parse(model_id: []const u8) ?ParseResult {
    const separator = std.mem.indexOf(u8, model_id, "::") orelse return null;
    const prefix = model_id[0..separator];
    const rest = model_id[separator + 2 ..];

    // "native::" is not a remote provider
    if (std.mem.eql(u8, prefix, "native")) {
        return null;
    }

    // Look up the provider
    if (getByName(prefix)) |provider| {
        return ParseResult{
            .provider = provider,
            .model_id = rest,
        };
    }

    return null;
}

/// Check if a model ID has a known provider prefix.
pub fn hasProviderPrefix(model_id: []const u8) bool {
    return parse(model_id) != null;
}

/// Check if a prefix string (e.g., "vllm::") is a known provider.
pub fn isProviderPrefix(prefix: []const u8) bool {
    if (!std.mem.endsWith(u8, prefix, "::")) {
        return false;
    }
    const name = prefix[0 .. prefix.len - 2];
    return getByName(name) != null;
}

/// Get API key from environment for a provider.
/// Returns null if no API key env var is configured or the env var is not set.
pub fn getApiKeyFromEnv(allocator: std.mem.Allocator, provider: Provider) !?[]u8 {
    const env_name = provider.api_key_env orelse return null;
    return std.process.getEnvVarOwned(allocator, env_name) catch |err| switch (err) {
        error.EnvironmentVariableNotFound => return null,
        else => return err,
    };
}

// =============================================================================
// Tests
// =============================================================================

test "getByName: known providers" {
    try std.testing.expect(getByName("vllm") != null);
    try std.testing.expect(getByName("ollama") != null);
    try std.testing.expect(getByName("llamacpp") != null);
    try std.testing.expect(getByName("lmstudio") != null);
    try std.testing.expect(getByName("localai") != null);
    try std.testing.expect(getByName("openai") != null);
    try std.testing.expect(getByName("openrouter") != null);
}

test "getByName: unknown provider" {
    try std.testing.expect(getByName("unknown") == null);
    try std.testing.expect(getByName("native") == null);
    try std.testing.expect(getByName("anthropic") == null);
    try std.testing.expect(getByName("bedrock") == null);
}

test "getByName: provider details" {
    const vllm = getByName("vllm").?;
    try std.testing.expectEqualStrings("vllm", vllm.name);
    try std.testing.expectEqualStrings("http://localhost:8000/v1", vllm.default_endpoint);
    try std.testing.expectEqualStrings("VLLM_API_KEY", vllm.api_key_env.?);

    const ollama = getByName("ollama").?;
    try std.testing.expectEqualStrings("http://localhost:11434/v1", ollama.default_endpoint);
    try std.testing.expect(ollama.api_key_env == null);

    const openai = getByName("openai").?;
    try std.testing.expectEqualStrings("https://api.openai.com/v1", openai.default_endpoint);
    try std.testing.expectEqualStrings("OPENAI_API_KEY", openai.api_key_env.?);

    const openrouter = getByName("openrouter").?;
    try std.testing.expectEqualStrings("https://openrouter.ai/api/v1", openrouter.default_endpoint);
    try std.testing.expectEqualStrings("OPENROUTER_API_KEY", openrouter.api_key_env.?);
}

test "parse: provider-prefixed model IDs" {
    const result1 = parse("vllm::org/model-name").?;
    try std.testing.expectEqualStrings("vllm", result1.provider.name);
    try std.testing.expectEqualStrings("org/model-name", result1.model_id);

    const result2 = parse("openai::gpt-4o").?;
    try std.testing.expectEqualStrings("openai", result2.provider.name);
    try std.testing.expectEqualStrings("gpt-4o", result2.model_id);

    const result3 = parse("ollama::model-name:latest").?;
    try std.testing.expectEqualStrings("ollama", result3.provider.name);
    try std.testing.expectEqualStrings("model-name:latest", result3.model_id);

    const result4 = parse("openrouter::openai/gpt-4o").?;
    try std.testing.expectEqualStrings("openrouter", result4.provider.name);
    try std.testing.expectEqualStrings("openai/gpt-4o", result4.model_id);
}

test "parse: no provider prefix" {
    try std.testing.expect(parse("org/model-name") == null);
    try std.testing.expect(parse("/path/to/model") == null);
    try std.testing.expect(parse("./local-model") == null);
}

test "parse: native prefix is not a provider" {
    try std.testing.expect(parse("native::org/model-name") == null);
}

test "parse: unknown provider" {
    try std.testing.expect(parse("anthropic::claude-3") == null);
    try std.testing.expect(parse("bedrock::model-id") == null);
    try std.testing.expect(parse("unknown::model") == null);
}

test "hasProviderPrefix" {
    try std.testing.expect(hasProviderPrefix("vllm::model"));
    try std.testing.expect(hasProviderPrefix("openai::gpt-4o"));
    try std.testing.expect(!hasProviderPrefix("org/model-name"));
    try std.testing.expect(!hasProviderPrefix("native::model"));
    try std.testing.expect(!hasProviderPrefix("unknown::model"));
}

test "isProviderPrefix" {
    try std.testing.expect(isProviderPrefix("vllm::"));
    try std.testing.expect(isProviderPrefix("openai::"));
    try std.testing.expect(isProviderPrefix("openrouter::"));
    try std.testing.expect(!isProviderPrefix("vllm"));
    try std.testing.expect(!isProviderPrefix("native::"));
    try std.testing.expect(!isProviderPrefix("unknown::"));
}

test "getApiKeyFromEnv: returns null when env var not set for vllm" {
    const vllm = getByName("vllm").?;
    // VLLM_API_KEY is unlikely to be set in test environment
    const result = try getApiKeyFromEnv(std.testing.allocator, vllm);
    if (result) |key| {
        // If it happens to be set, just free it
        std.testing.allocator.free(key);
    }
}

test "getApiKeyFromEnv: returns null when env var not set" {
    const provider = Provider{
        .name = "test",
        .default_endpoint = "http://localhost:9999",
        .api_key_env = "TALU_TEST_NONEXISTENT_VAR_12345",
    };
    const result = try getApiKeyFromEnv(std.testing.allocator, provider);
    try std.testing.expect(result == null);
}

test "getApiKeyFromEnv: reads PATH env var as proof of concept" {
    // PATH is guaranteed to exist on all systems - use it to verify
    // the function actually reads environment variables correctly.
    const provider = Provider{
        .name = "test",
        .default_endpoint = "http://localhost:9999",
        .api_key_env = "PATH",
    };
    const result = try getApiKeyFromEnv(std.testing.allocator, provider);
    try std.testing.expect(result != null);
    defer std.testing.allocator.free(result.?);

    // PATH should contain at least one path separator
    try std.testing.expect(result.?.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, result.?, "/") != null or
        std.mem.indexOf(u8, result.?, "\\") != null);
}
