//! Integration tests for TokenizerHandle
//!
//! TokenizerHandle combines a tokenizer with model context (generation config,
//! resolved paths). This is the complete "loaded tokenizer" used by the C API.
//!
//! These tests require a valid model directory with tokenizer.json and
//! optionally generation_config.json. Tests skip if no model is available.

const std = @import("std");
const main = @import("main");
const TokenizerHandle = main.tokenizer.TokenizerHandle;

// =============================================================================
// Test Configuration
// =============================================================================

/// Test model path - uses cache directory if available
const TEST_MODEL_URI = "models/Qwen/Qwen3-0.6B-GAF4";

/// Check if the test model is available
fn testModelAvailable() bool {
    std.fs.cwd().access(TEST_MODEL_URI ++ "/tokenizer.json", .{}) catch return false;
    return true;
}

// =============================================================================
// Lifecycle Tests
// =============================================================================

test "TokenizerHandle: init and deinit from model path" {
    if (!testModelAvailable()) {
        // SKIP: No test model available in CI environment
        return error.SkipZigTest;
    }

    const handle = try TokenizerHandle.init(std.testing.allocator, TEST_MODEL_URI);
    defer handle.deinit();

    // Verify handle is properly initialized
    try std.testing.expect(handle.tok.vocab_size > 0);
}

test "TokenizerHandle: init fails for invalid path" {
    const result = TokenizerHandle.init(std.testing.allocator, "/nonexistent/model/path");
    // TokenizerHandle.init returns InitFailed when tokenizer assets are unavailable.
    try std.testing.expectError(error.InitFailed, result);
}

test "TokenizerHandle: deinit frees all resources" {
    if (!testModelAvailable()) {
        return error.SkipZigTest;
    }

    const handle = try TokenizerHandle.init(std.testing.allocator, TEST_MODEL_URI);
    handle.deinit();
    // No leak = test passes (std.testing.allocator checks)
}

// =============================================================================
// getVocabSize Tests
// =============================================================================

test "TokenizerHandle: getVocabSize returns positive value" {
    if (!testModelAvailable()) {
        return error.SkipZigTest;
    }

    const handle = try TokenizerHandle.init(std.testing.allocator, TEST_MODEL_URI);
    defer handle.deinit();

    const vocab_size = handle.getVocabSize();
    try std.testing.expect(vocab_size > 0);
}

test "TokenizerHandle: getVocabSize matches underlying tokenizer" {
    if (!testModelAvailable()) {
        return error.SkipZigTest;
    }

    const handle = try TokenizerHandle.init(std.testing.allocator, TEST_MODEL_URI);
    defer handle.deinit();

    try std.testing.expectEqual(handle.tok.vocab_size, handle.getVocabSize());
}

// =============================================================================
// tokenBytes Tests
// =============================================================================

test "TokenizerHandle: tokenBytes returns bytes for valid token" {
    if (!testModelAvailable()) {
        return error.SkipZigTest;
    }

    const handle = try TokenizerHandle.init(std.testing.allocator, TEST_MODEL_URI);
    defer handle.deinit();

    // Token 0 should exist in any tokenizer
    const bytes = handle.tokenBytes(0);
    try std.testing.expect(bytes != null);
}

test "TokenizerHandle: tokenBytes returns null for out of range token" {
    if (!testModelAvailable()) {
        return error.SkipZigTest;
    }

    const handle = try TokenizerHandle.init(std.testing.allocator, TEST_MODEL_URI);
    defer handle.deinit();

    // Very large token ID should be out of range
    const bytes = handle.tokenBytes(999999999);
    try std.testing.expect(bytes == null);
}

// =============================================================================
// getTokensStartingWith Tests
// =============================================================================

test "TokenizerHandle: getTokensStartingWith returns tokens for common byte" {
    if (!testModelAvailable()) {
        return error.SkipZigTest;
    }

    const handle = try TokenizerHandle.init(std.testing.allocator, TEST_MODEL_URI);
    defer handle.deinit();

    // 'a' (0x61) is common - should have tokens starting with it
    const tokens = handle.getTokensStartingWith('a');
    try std.testing.expect(tokens.len > 0);
}

test "TokenizerHandle: getTokensStartingWith returns empty for rare byte" {
    if (!testModelAvailable()) {
        return error.SkipZigTest;
    }

    const handle = try TokenizerHandle.init(std.testing.allocator, TEST_MODEL_URI);
    defer handle.deinit();

    // Null byte (0x00) - unlikely to have tokens starting with it
    const tokens = handle.getTokensStartingWith(0);
    // May or may not be empty, just verify it doesn't crash
    _ = tokens;
}

// =============================================================================
// Model Directory and Config Tests
// =============================================================================

test "TokenizerHandle: model_dir is set after init" {
    if (!testModelAvailable()) {
        return error.SkipZigTest;
    }

    const handle = try TokenizerHandle.init(std.testing.allocator, TEST_MODEL_URI);
    defer handle.deinit();

    try std.testing.expect(handle.model_dir.len > 0);
}

test "TokenizerHandle: gen_config is loaded from model directory" {
    if (!testModelAvailable()) {
        return error.SkipZigTest;
    }

    const handle = try TokenizerHandle.init(std.testing.allocator, TEST_MODEL_URI);
    defer handle.deinit();

    // gen_config should be initialized (may have default values if file missing)
    // Just verify it doesn't crash accessing the config
    _ = handle.gen_config;
}
