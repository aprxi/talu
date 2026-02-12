//! Integration tests for io.repository.ModelSource
//!
//! ModelSource provides a unified interface for checking model existence and
//! listing files across different remote storage backends (HuggingFace Hub, S3).

const std = @import("std");
const main = @import("main");
const ModelSource = main.io.repository.ModelSource;
const SourceConfig = main.io.repository.SourceConfig;
const SourceError = main.io.repository.SourceError;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "ModelSource type is accessible" {
    const T = ModelSource;
    _ = T;
}

test "ModelSource is a tagged union" {
    const info = @typeInfo(ModelSource);
    try std.testing.expect(info == .@"union");
}

test "ModelSource has huggingface variant" {
    const src = ModelSource{ .huggingface = {} };
    try std.testing.expect(src == .huggingface);
}

test "ModelSource has s3 variant" {
    const S3Source = @typeInfo(ModelSource).@"union".fields[1].type;
    _ = S3Source;
}

test "ModelSource.default is huggingface" {
    const src = ModelSource.default;
    try std.testing.expect(src == .huggingface);
}

// =============================================================================
// SourceConfig Tests
// =============================================================================

test "SourceConfig type is accessible" {
    const T = SourceConfig;
    _ = T;
}

test "SourceConfig is a struct" {
    const info = @typeInfo(SourceConfig);
    try std.testing.expect(info == .@"struct");
}

test "SourceConfig has token field" {
    const config = SourceConfig{ .token = null };
    try std.testing.expect(config.token == null);

    const with_token = SourceConfig{ .token = "test_token" };
    try std.testing.expect(with_token.token != null);
}

// =============================================================================
// SourceError Tests
// =============================================================================

test "SourceError type is accessible" {
    const T = SourceError;
    _ = T;
}

test "SourceError is an error set" {
    const info = @typeInfo(SourceError);
    try std.testing.expect(info == .error_set);
}

test "SourceError contains expected errors" {
    // Verify key error variants exist by creating them
    const errors = [_]SourceError{
        SourceError.NetworkError,
        SourceError.ParseError,
        SourceError.Unauthorized,
        SourceError.RateLimited,
        SourceError.NotSupported,
        SourceError.OutOfMemory,
    };
    _ = errors;
}

// =============================================================================
// Method Existence Tests
// =============================================================================

test "ModelSource has exists method" {
    try std.testing.expect(@hasDecl(ModelSource, "exists"));
}

test "ModelSource has listFiles method" {
    try std.testing.expect(@hasDecl(ModelSource, "listFiles"));
}

// =============================================================================
// S3Source Tests (Future Implementation)
// =============================================================================

test "S3Source returns NotSupported for exists" {
    const allocator = std.testing.allocator;
    const s3_source = main.io.repository.source.S3Source{ .bucket = "test-bucket" };
    const result = s3_source.exists(allocator, "test/model", .{});
    try std.testing.expectError(SourceError.NotSupported, result);
}

test "S3Source returns NotSupported for listFiles" {
    const allocator = std.testing.allocator;
    const s3_source = main.io.repository.source.S3Source{ .bucket = "test-bucket" };
    const result = s3_source.listFiles(allocator, "test/model", .{});
    try std.testing.expectError(SourceError.NotSupported, result);
}
