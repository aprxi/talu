//! Scheme Parsing for Model Repositories
//!
//! Parses URIs to determine WHERE a repository-based model is stored.
//! This is called by resolveModelPath() after the Router has determined
//! the model is repository-based (not an external API).
//!
//! ## Scope
//!
//! This module ONLY handles repository storage schemes:
//!   - Local filesystem paths
//!   - Bare model IDs (e.g., `org/model-name`) - default to HuggingFace Hub
//!
//! External API prefixes (`openai:`, `anthropic:`, `bedrock:`) are handled
//! by the C API layer (capi/router.zig) BEFORE reaching this module.
//!
//! ## Supported Schemes
//!
//! | Scheme   | Example                    | Storage         |
//! |----------|----------------------------|-----------------|
//! | `.local` | `/path`, `./path`, `~/path`| Local filesystem|
//! | `.hub`   | `org/model-name`           | HF Hub (default)|
//!
//! ## Resolution Logic
//!
//! 1. Reject unknown schemes (anything://...)
//! 2. Check for filesystem path indicators (`/`, `./`, `../`, `~/`, `C:/`)
//! 3. Check if path exists as local directory
//! 4. Default to `.hub` (HuggingFace Hub - fetch if not cached)
//!
//! ## S3 / Custom Endpoints
//!
//! S3-compatible storage (MinIO, R2, etc.) is supported via HF_ENDPOINT
//! environment variable or `endpoint_url` parameter, not via URI scheme.

const std = @import("std");
const cache = @import("cache.zig");

/// URI scheme for model sources
pub const Scheme = enum {
    /// Local filesystem path
    local,
    /// HuggingFace Hub (default for bare model IDs like "org/model-name")
    /// Checks cache first, then fetches from hub if not found
    hub,
};

/// Errors that can occur during URI parsing.
pub const ParseError = error{
    /// Unknown URI scheme (e.g., xyz://)
    UnknownScheme,
};

/// Parsed URI with scheme and path
pub const Uri = struct {
    scheme: Scheme,
    path: []const u8,
};

/// Parse a URI string into scheme and path.
///
/// Resolution order:
/// 1. Reject unknown schemes (anything://...)
/// 2. Filesystem paths: starts with /, ./, ../
/// 3. Directory check: if path exists as directory → local
/// 4. Default: HuggingFace Hub (cache first, then fetch)
pub fn parse(uri: []const u8) ParseError!Uri {
    // Empty string → hub (will list all)
    if (uri.len == 0) {
        return .{ .scheme = .hub, .path = uri };
    }

    // Reject unknown schemes (anything with "://")
    if (std.mem.indexOf(u8, uri, "://")) |_| {
        return ParseError.UnknownScheme;
    }

    // Filesystem path indicators
    if (std.mem.startsWith(u8, uri, "/")) {
        return .{ .scheme = .local, .path = uri };
    }
    if (std.mem.startsWith(u8, uri, "./")) {
        return .{ .scheme = .local, .path = uri };
    }
    if (std.mem.startsWith(u8, uri, "../")) {
        return .{ .scheme = .local, .path = uri };
    }
    // Home directory expansion
    if (std.mem.startsWith(u8, uri, "~/")) {
        return .{ .scheme = .local, .path = uri };
    }
    // Windows absolute paths
    if (uri.len > 1 and uri[1] == ':') {
        return .{ .scheme = .local, .path = uri };
    }

    // Check if it's an existing directory
    if (isDirectory(uri)) {
        return .{ .scheme = .local, .path = uri };
    }

    // Default: HuggingFace Hub (checks cache first, fetches if not found)
    return .{ .scheme = .hub, .path = uri };
}

/// Check if a path is an existing directory.
fn isDirectory(path: []const u8) bool {
    const stat = std.fs.cwd().statFile(path) catch return false;
    return stat.kind == .directory;
}

/// Check if a URI refers to a remote source (may require network).
pub fn isRemote(uri: Uri) bool {
    return uri.scheme == .hub;
}

/// Check if a URI refers to a local source (no network needed).
pub fn isLocal(uri: Uri) bool {
    return uri.scheme == .local;
}

// =============================================================================
// Tests
// =============================================================================

test "parse absolute path" {
    const result = try parse("/home/user/models/my-model");
    try std.testing.expectEqual(Scheme.local, result.scheme);
    try std.testing.expectEqualStrings("/home/user/models/my-model", result.path);
}

test "parse relative path ./" {
    const result = try parse("./my-model");
    try std.testing.expectEqual(Scheme.local, result.scheme);
    try std.testing.expectEqualStrings("./my-model", result.path);
}

test "parse relative path ../" {
    const result = try parse("../other-model");
    try std.testing.expectEqual(Scheme.local, result.scheme);
    try std.testing.expectEqualStrings("../other-model", result.path);
}

test "parse home directory path" {
    const result = try parse("~/models/my-model");
    try std.testing.expectEqual(Scheme.local, result.scheme);
    try std.testing.expectEqualStrings("~/models/my-model", result.path);
}

test "parse Windows path" {
    const result = try parse("C:/models/my-model");
    try std.testing.expectEqual(Scheme.local, result.scheme);
    try std.testing.expectEqualStrings("C:/models/my-model", result.path);
}

test "parse model ID defaults to hub" {
    // Non-existent path that looks like model ID → HuggingFace Hub
    const result = try parse("org/model-name");
    try std.testing.expectEqual(Scheme.hub, result.scheme);
    try std.testing.expectEqualStrings("org/model-name", result.path);
}

test "parse empty string" {
    const result = try parse("");
    try std.testing.expectEqual(Scheme.hub, result.scheme);
    try std.testing.expectEqualStrings("", result.path);
}

test "parse unknown scheme returns error" {
    try std.testing.expectError(ParseError.UnknownScheme, parse("hf://org/model-name"));
    try std.testing.expectError(ParseError.UnknownScheme, parse("s3://bucket/path"));
    try std.testing.expectError(ParseError.UnknownScheme, parse("xyz://something"));
    try std.testing.expectError(ParseError.UnknownScheme, parse("http://example.com"));
}

test "isRemote" {
    try std.testing.expect(isRemote(.{ .scheme = .hub, .path = "x" }));
    try std.testing.expect(!isRemote(.{ .scheme = .local, .path = "x" }));
}

test "isLocal" {
    try std.testing.expect(!isLocal(.{ .scheme = .hub, .path = "x" }));
    try std.testing.expect(isLocal(.{ .scheme = .local, .path = "x" }));
}
