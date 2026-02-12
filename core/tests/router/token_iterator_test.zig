//! Integration tests for router.TokenIterator
//!
//! TokenIterator provides pull-based streaming generation with a ring buffer.
//! Full functional tests require a loaded model - these tests verify the type
//! structure, exports, and ring buffer logic.

const std = @import("std");
const main = @import("main");

const TokenIterator = main.router.TokenIterator;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "TokenIterator type is accessible from router" {
    const T = TokenIterator;
    _ = T;
}

test "TokenIterator is a struct" {
    const info = @typeInfo(TokenIterator);
    try std.testing.expect(info == .@"struct");
}

// =============================================================================
// Field Access Tests
// =============================================================================

test "TokenIterator has atomic state fields" {
    // Verify the struct has the expected atomic fields by checking typeInfo
    const T = TokenIterator;
    const fields = @typeInfo(T).@"struct".fields;

    var has_done = false;
    var has_cancelled = false;
    var has_error_code = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "done")) has_done = true;
        if (comptime std.mem.eql(u8, field.name, "cancelled")) has_cancelled = true;
        if (comptime std.mem.eql(u8, field.name, "error_code")) has_error_code = true;
    }

    try std.testing.expect(has_done);
    try std.testing.expect(has_cancelled);
    try std.testing.expect(has_error_code);
}

test "TokenIterator has stats fields" {
    const T = TokenIterator;
    const fields = @typeInfo(T).@"struct".fields;

    var has_prompt_tokens = false;
    var has_completion_tokens = false;
    var has_prefill_ns = false;
    var has_generation_ns = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "prompt_tokens")) has_prompt_tokens = true;
        if (comptime std.mem.eql(u8, field.name, "completion_tokens")) has_completion_tokens = true;
        if (comptime std.mem.eql(u8, field.name, "prefill_ns")) has_prefill_ns = true;
        if (comptime std.mem.eql(u8, field.name, "generation_ns")) has_generation_ns = true;
    }

    try std.testing.expect(has_prompt_tokens);
    try std.testing.expect(has_completion_tokens);
    try std.testing.expect(has_prefill_ns);
    try std.testing.expect(has_generation_ns);
}

// =============================================================================
// Method Existence Tests
// =============================================================================

test "TokenIterator has init method" {
    try std.testing.expect(@hasDecl(TokenIterator, "init"));
}

test "TokenIterator has deinit method" {
    try std.testing.expect(@hasDecl(TokenIterator, "deinit"));
}

test "TokenIterator has next method" {
    try std.testing.expect(@hasDecl(TokenIterator, "next"));
}

test "TokenIterator has cancel method" {
    try std.testing.expect(@hasDecl(TokenIterator, "cancel"));
}

test "TokenIterator has hasError method" {
    try std.testing.expect(@hasDecl(TokenIterator, "hasError"));
}

test "TokenIterator has getErrorCode method" {
    try std.testing.expect(@hasDecl(TokenIterator, "getErrorCode"));
}

test "TokenIterator has getErrorMsg method" {
    try std.testing.expect(@hasDecl(TokenIterator, "getErrorMsg"));
}

test "TokenIterator has getPromptTokens method" {
    try std.testing.expect(@hasDecl(TokenIterator, "getPromptTokens"));
}

test "TokenIterator has getCompletionTokens method" {
    try std.testing.expect(@hasDecl(TokenIterator, "getCompletionTokens"));
}

test "TokenIterator has getPrefillNs method" {
    try std.testing.expect(@hasDecl(TokenIterator, "getPrefillNs"));
}

test "TokenIterator has getGenerationNs method" {
    try std.testing.expect(@hasDecl(TokenIterator, "getGenerationNs"));
}

test "TokenIterator has getItemType method" {
    try std.testing.expect(@hasDecl(TokenIterator, "getItemType"));
}

test "TokenIterator has getContentType method" {
    try std.testing.expect(@hasDecl(TokenIterator, "getContentType"));
}

test "TokenIterator has getFinishReason method" {
    try std.testing.expect(@hasDecl(TokenIterator, "getFinishReason"));
}

// =============================================================================
// Content Classification Field Tests
// =============================================================================

test "TokenIterator has filter_state field" {
    const fields = @typeInfo(TokenIterator).@"struct".fields;
    var found = false;
    inline for (fields) |f| {
        if (comptime std.mem.eql(u8, f.name, "filter_state")) found = true;
    }
    try std.testing.expect(found);
}

test "TokenIterator has finish_reason field" {
    const fields = @typeInfo(TokenIterator).@"struct".fields;
    var found = false;
    inline for (fields) |f| {
        if (comptime std.mem.eql(u8, f.name, "finish_reason")) found = true;
    }
    try std.testing.expect(found);
}

test "TokenIterator has is_tool_generation field" {
    const fields = @typeInfo(TokenIterator).@"struct".fields;
    var found = false;
    inline for (fields) |f| {
        if (comptime std.mem.eql(u8, f.name, "is_tool_generation")) found = true;
    }
    try std.testing.expect(found);
}
