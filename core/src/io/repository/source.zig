//! Source Abstraction for Model Repositories
//!
//! Provides a unified interface for checking model existence and listing files
//! from HuggingFace Hub.
//!
//! ## Design
//!
//! The Source abstraction separates "where to fetch from" from "how to cache".
//! All sources share the same local HF-style cache (~/.cache/huggingface/hub/).
//!
//! ```
//! User: repo.exists("org/model-name")
//!          ↓
//!       1. Check local cache (cache.isCached)
//!          ↓ (miss)
//!       2. Check source (Source.exists)
//!          ↓
//!       HuggingFace API
//! ```
//!
//! ## S3-Compatible Endpoints
//!
//! For S3-compatible storage (MinIO, R2, etc.), use the HF_ENDPOINT
//! environment variable or `endpoint_url` parameter instead of a URI scheme.
//!

const std = @import("std");
const http = @import("../transport/http.zig");
const hf = @import("../transport/hf.zig");

/// Errors that can occur during source operations.
pub const SourceError = error{
    /// Network request failed
    NetworkError,
    /// API response could not be parsed
    ParseError,
    /// Authentication failed (invalid/missing token)
    Unauthorized,
    /// Rate limited by the source
    RateLimited,
    /// Source type not yet implemented
    NotSupported,
    /// Out of memory
    OutOfMemory,
    /// Unexpected error
    Unexpected,
};

/// Configuration for source operations.
pub const SourceConfig = struct {
    /// Authentication token (optional, source-specific)
    token: ?[]const u8 = null,
    /// Custom HF endpoint URL (optional, overrides HF_ENDPOINT env var)
    endpoint_url: ?[]const u8 = null,
};

/// HuggingFace Hub source.
///
/// Checks model existence and lists files via the HuggingFace API.
pub const HuggingFaceSource = struct {
    const DEFAULT_HF_ENDPOINT = "https://huggingface.co";

    /// Get effective endpoint URL (config > env > default)
    fn getEffectiveEndpoint(config_endpoint: ?[]const u8) []const u8 {
        if (config_endpoint) |endpoint| return endpoint;
        if (std.posix.getenv("HF_ENDPOINT")) |env_endpoint| {
            return std.mem.sliceTo(env_endpoint, 0);
        }
        return DEFAULT_HF_ENDPOINT;
    }

    /// Check if a model exists on HuggingFace Hub.
    ///
    /// Makes an HTTP request to the HF API. Returns true if the model
    /// exists and is accessible (considering auth token if provided).
    pub fn exists(allocator: std.mem.Allocator, model_id: []const u8, config: SourceConfig) SourceError!bool {
        const base_url = getEffectiveEndpoint(config.endpoint_url);
        const api_url = std.fmt.allocPrint(allocator, "{s}/api/models/{s}", .{ base_url, model_id }) catch {
            return SourceError.OutOfMemory;
        };
        defer allocator.free(api_url);

        // Attempt to fetch model info - if successful, model exists
        const response = http.fetch(allocator, api_url, .{ .token = config.token }) catch |err| {
            return switch (err) {
                error.NotFound => false,
                error.Unauthorized => SourceError.Unauthorized,
                error.RateLimited => SourceError.RateLimited,
                error.ResponseTooLarge => SourceError.NetworkError,
                error.OutOfMemory => SourceError.OutOfMemory,
                else => SourceError.NetworkError,
            };
        };
        allocator.free(response);
        return true;
    }

    /// List files in a model repository on HuggingFace Hub.
    ///
    /// Returns an owned slice of owned strings. Caller must free both
    /// the strings and the slice.
    pub fn listFiles(allocator: std.mem.Allocator, model_id: []const u8, config: SourceConfig) SourceError![][]const u8 {
        const file_list = hf.fetchFileList(allocator, model_id, .{
            .token = config.token,
            .endpoint_url = config.endpoint_url,
        }) catch |err| {
            return switch (err) {
                error.NotFound => SourceError.NetworkError,
                error.Unauthorized => SourceError.Unauthorized,
                error.RateLimited => SourceError.RateLimited,
                error.ResponseTooLarge => SourceError.NetworkError,
                error.OutOfMemory => SourceError.OutOfMemory,
                error.ApiResponseParseError => SourceError.ParseError,
                else => SourceError.NetworkError,
            };
        };
        return file_list;
    }
};

/// Unified source type for model repositories.
///
/// Provides a common interface for fetching models from HuggingFace Hub.
/// For S3-compatible endpoints, use HF_ENDPOINT or endpoint_url parameter.
///
/// Example:
/// ```zig
/// const src = ModelSource.default;  // HuggingFace Hub
/// const exists = try src.exists(allocator, "org/model-name", .{});
/// ```
pub const ModelSource = union(enum) {
    /// HuggingFace Hub (default)
    huggingface: void,

    /// Default source: HuggingFace Hub
    pub const default: ModelSource = .{ .huggingface = {} };

    /// Check if a model exists on this source.
    ///
    /// This is a network operation - it queries the remote source.
    /// For cache-only checks, use cache.isCached() directly.
    pub fn exists(self: ModelSource, allocator: std.mem.Allocator, model_id: []const u8, config: SourceConfig) SourceError!bool {
        return switch (self) {
            .huggingface => HuggingFaceSource.exists(allocator, model_id, config),
        };
    }

    /// List files in a model on this source.
    ///
    /// Returns filenames available in the model repository.
    /// Caller owns returned memory.
    pub fn listFiles(self: ModelSource, allocator: std.mem.Allocator, model_id: []const u8, config: SourceConfig) SourceError![][]const u8 {
        return switch (self) {
            .huggingface => HuggingFaceSource.listFiles(allocator, model_id, config),
        };
    }
};

// =============================================================================
// Tests
// =============================================================================

test "ModelSource.default is huggingface" {
    const src = ModelSource.default;
    try std.testing.expect(src == .huggingface);
}
