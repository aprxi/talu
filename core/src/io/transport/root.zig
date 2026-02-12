//! Transport Layer
//!
//! Handles fetching model files from remote sources:
//! - HuggingFace Hub API
//! - Generic HTTP client
//! - Future: S3, other cloud storage
//!
//! The transport layer is separate from the repository format layer.
//! Repository understands the structure, transport fetches the files.

const std = @import("std");

/// Generic HTTP client (libcurl wrapper)
pub const http = @import("http.zig");

/// HuggingFace Hub API
pub const hf = @import("hf.zig");

// Re-export commonly used types
pub const HttpConfig = http.HttpConfig;
pub const HttpError = http.HttpError;
pub const ProgressCallback = http.ProgressCallback;
pub const FileStartCallback = http.FileStartCallback;

pub const DownloadConfig = hf.DownloadConfig;
pub const DownloadError = hf.DownloadError;

/// Initialize HTTP globally (call once at program start)
pub const globalInit = http.globalInit;

/// Clean up HTTP globally (call once at program end)
pub const globalCleanup = http.globalCleanup;

// =============================================================================
// Tests
// =============================================================================

test "transport module compiles" {
    _ = http;
    _ = hf;
}
