//! Source Abstraction for Model Repositories
//!
//! Provides a unified interface for listing files from HuggingFace Hub.
//!
//! ## Design
//!
//! The Source abstraction separates "where to fetch from" from "how to cache".
//! All sources share the same local HF-style cache (~/.cache/huggingface/hub/).
//!
//! ## S3-Compatible Endpoints
//!
//! For S3-compatible storage (MinIO, R2, etc.), use the HF_ENDPOINT
//! environment variable or `endpoint_url` parameter instead of a URI scheme.
//!

const std = @import("std");
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
/// const files = try src.listFiles(allocator, "org/model-name", .{});
/// ```
pub const ModelSource = union(enum) {
    /// HuggingFace Hub (default)
    huggingface: void,

    /// Default source: HuggingFace Hub
    pub const default: ModelSource = .{ .huggingface = {} };

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
