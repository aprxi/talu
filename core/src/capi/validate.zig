//! C API for structured output validation.
//!
//! This module provides three APIs:
//!
//! 1. Chat response-format API (talu_set_response_format*):
//!    - Attaches JSON-schema-constrained generation to chat sessions
//!    - Includes thinking mode, stop tokens, and reusable grammar handles
//!
//! 2. Low-level engine API (talu_validate_engine_*):
//!    - Direct access to validation state machine
//!    - Byte-level operations
//!    - Useful for benchmarking, validation, and standalone use cases
//!
//! 3. Grammar handle API (talu_grammar_*):
//!    - Pre-compile schemas into reusable grammar handles
//!    - Zero-latency structured output when reusing compiled grammars
//!    - Use with talu_set_response_format_handle for optimal performance

const std = @import("std");
const validate_mod = @import("validate_pkg");
const sampler_mod = validate_mod.sampler;
const cache_mod = validate_mod.cache;
const responses_capi = @import("responses.zig");
const responses_mod = @import("../responses/root.zig");
const capi_error = @import("error.zig");
const error_codes = @import("error_codes.zig");

const ConstrainedSampler = sampler_mod.ConstrainedSampler;
const GrammarConfig = sampler_mod.GrammarConfig;
const ChatHandle = responses_capi.ChatHandle;
const Chat = responses_mod.Chat;
const Validator = validate_mod.Validator;
const Grammar = validate_mod.ast.Grammar;

/// Configuration for structured output validation.
pub const ValidateConfigC = extern struct {
    allow_thinking: bool = false,
    max_thinking_tokens: usize = 512,
    start_marker: ?[*:0]const u8 = null,
    soft_limit_ratio: f32 = 0.9,
    soft_limit_bias: f32 = -2.0,
};

/// Sets the response format (JSON schema) for a chat session.
///
/// Subsequent generation will be constrained to produce valid JSON matching the schema.
///
/// Parameters:
///   chat_handle: Chat session handle
///   schema_json: JSON Schema string
///   config: Optional configuration (thinking mode, soft limits)
///   stop_tokens: EOS token IDs
///   stop_tokens_len: Number of stop tokens
///   prefix_token_ids: Optional prefix tokens to prepend
///   prefix_token_ids_len: Number of prefix tokens
///
/// Returns 0 on success, negative error code on failure.
pub export fn talu_set_response_format(
    chat_handle: ?*ChatHandle,
    schema_json: [*:0]const u8,
    config: ?*const ValidateConfigC,
    stop_tokens: ?[*]const u32,
    stop_tokens_len: usize,
    prefix_token_ids: ?[*]const u32,
    prefix_token_ids_len: usize,
) callconv(.c) i32 {
    capi_error.clearError();

    // Cast opaque handle to Chat
    const chat_state: *Chat = @ptrCast(@alignCast(chat_handle orelse {
        capi_error.setError(error.InvalidHandle, "Invalid chat handle", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));

    var grammar_config = GrammarConfig{};
    if (config) |cfg| {
        grammar_config = .{
            .allow_thinking = cfg.allow_thinking,
            .max_thinking_tokens = cfg.max_thinking_tokens,
            .start_marker = if (cfg.start_marker) |m| std.mem.span(m) else null,
            .soft_limit_ratio = cfg.soft_limit_ratio,
            .soft_limit_bias = cfg.soft_limit_bias,
        };
    }

    const stop_slice = if (stop_tokens) |ptr| ptr[0..stop_tokens_len] else &[_]u32{};
    const prefix_slice = if (prefix_token_ids) |ptr| ptr[0..prefix_token_ids_len] else &[_]u32{};

    setResponseFormat(
        chat_state,
        schema_json,
        grammar_config,
        stop_slice,
        prefix_slice,
    ) catch |err| {
        if (err == error.AllocFailed) {
            capi_error.setError(error.OutOfMemory, "Failed to allocate sampler", .{});
            return @intFromEnum(error_codes.ErrorCode.out_of_memory);
        }
        capi_error.setError(err, "Failed to parse schema", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Clears the response format from a chat session.
///
/// Subsequent generation will be unconstrained.
pub export fn talu_clear_response_format(chat_handle: ?*ChatHandle) callconv(.c) void {
    if (chat_handle) |handle| {
        // Cast opaque handle to Chat
        const chat_state: *Chat = @ptrCast(@alignCast(handle));
        chat_state.clearGrammar();
    }
}

/// C-compatible result for semantic validation.
pub const SemanticValidationResultC = extern struct {
    /// True if validation passed, false if there was a violation.
    is_valid: bool,
    /// If is_valid is false, this points to the JSON path of the violation.
    /// Null-terminated. Valid until next call or sampler destruction.
    path: ?[*:0]const u8,
    /// If is_valid is false, this points to a human-readable error message.
    /// Null-terminated. Valid until next call or sampler destruction.
    message: ?[*:0]const u8,
};

/// Validates the final generated output against semantic constraints.
///
/// Call this after generation completes to check constraints that grammar cannot
/// express: required fields, number ranges, additionalProperties violations.
///
/// Copy violation strings to C-compatible pointers. Leaks intentionally for C ABI.
fn copyViolationStrings(result: *SemanticValidationResultC, violation: anytype) void {
    const path_z = std.heap.c_allocator.dupeZ(u8, violation.path) catch {
        result.path = null;
        result.message = null;
        return;
    };
    const message_z = std.heap.c_allocator.dupeZ(u8, violation.message) catch {
        std.heap.c_allocator.free(path_z);
        result.path = null;
        result.message = null;
        return;
    };
    result.path = path_z.ptr;
    result.message = message_z.ptr;
}

/// The grammar ensures syntactically valid JSON, but cannot enforce all JSON Schema
/// constraints. This function performs post-generation validation of those constraints.
///
/// Parameters:
///   chat_handle: Chat session handle with active response format.
///   out_result: Output struct filled with validation result.
///
/// Returns:
///   0 if validation was performed (check out_result.is_valid for actual result).
///   Negative error code if validation could not be performed (no active grammar, etc.).
pub export fn talu_validate_response_format(
    chat_handle: ?*ChatHandle,
    out_result: ?*SemanticValidationResultC,
) callconv(.c) i32 {
    capi_error.clearError();

    const result = out_result orelse {
        capi_error.setError(error.InvalidArgument, "out_result is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    result.* = .{ .is_valid = true, .path = null, .message = null };

    const chat_state: *Chat = @ptrCast(@alignCast(chat_handle orelse {
        capi_error.setError(error.InvalidHandle, "Invalid chat handle", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));

    const sampler = chat_state.grammar_sampler orelse return 0;
    if (sampler.validateFinal()) |violation| {
        result.is_valid = false;
        copyViolationStrings(result, violation);
    }
    return 0;
}

// ============================================================================
// Pre-compiled Grammar Handle API
// ============================================================================
//
// This API allows pre-compiling JSON schemas into reusable grammar handles.
// Benefits:
// - Zero-latency structured output when reusing compiled grammars
// - Compile once, use many times
// - Schema validation happens at compile time, not at generation time

/// Internal struct holding pre-compiled grammar data.
const GrammarHandleData = struct {
    /// Copy of schema JSON (owned) - needed if grammar gets evicted from cache.
    schema_json: []u8,
    /// Cache key for O(1) grammar lookup.
    cache_key: cache_mod.CacheKey,
    allocator: std.mem.Allocator,

    fn deinit(self: *GrammarHandleData) void {
        self.allocator.free(self.schema_json);
        self.* = undefined;
    }
};

/// Opaque handle for pre-compiled grammar.
/// Thread safety: Thread-safe. Can be shared across threads.
/// The underlying grammar is stored in the global cache.
pub const GrammarHandle = opaque {
    fn fromPtr(ptr: *GrammarHandleData) *GrammarHandle {
        return @ptrCast(ptr);
    }

    fn toPtr(self: *GrammarHandle) *GrammarHandleData {
        return @ptrCast(@alignCast(self));
    }
};

/// Compiles a JSON schema into a reusable grammar handle.
///
/// The grammar is compiled once and cached for reuse. Passing the same schema
/// again will return quickly (cache lookup). Use this handle with
/// `talu_set_response_format_handle` for zero-latency structured output.
///
/// Parameters:
///   schema_json: JSON Schema string defining the output structure.
///
/// Returns handle on success, null on schema parse error or OOM.
/// Caller must call talu_grammar_free() when done.
///
/// Error behavior: On invalid schema, returns null and sets error message
/// retrievable via talu_get_last_error().
pub export fn talu_grammar_compile(
    schema_json: [*:0]const u8,
) callconv(.c) ?*GrammarHandle {
    capi_error.clearError();
    const allocator = std.heap.c_allocator;
    const schema_slice = std.mem.span(schema_json);

    // Compute cache key
    const cache_key = cache_mod.computeKey(schema_slice, .{});

    // Try to compile (will be cached if not already)
    const grammar_cache = cache_mod.getGlobalCache(allocator);
    _ = grammar_cache.getOrCompile(schema_slice, .{}) catch |err| {
        capi_error.setError(err, "Failed to compile schema", .{});
        return null;
    };

    // Allocate handle data
    const data = allocator.create(GrammarHandleData) catch {
        capi_error.setError(error.OutOfMemory, "Failed to allocate grammar handle", .{});
        return null;
    };
    errdefer allocator.destroy(data);

    // Copy schema JSON (needed if grammar gets evicted from cache)
    const schema_copy = allocator.dupe(u8, schema_slice) catch {
        capi_error.setError(error.OutOfMemory, "Failed to copy schema", .{});
        return null;
    };

    data.* = .{
        .schema_json = schema_copy,
        .cache_key = cache_key,
        .allocator = allocator,
    };

    return GrammarHandle.fromPtr(data);
}

/// Frees a pre-compiled grammar handle.
///
/// Safe to call with null (no-op).
/// The underlying grammar may remain in cache for other users.
pub export fn talu_grammar_free(handle: ?*GrammarHandle) callconv(.c) void {
    const data = (handle orelse return).toPtr();
    data.deinit();
    std.heap.c_allocator.destroy(data);
}

/// Sets the response format using a pre-compiled grammar handle.
///
/// This is faster than `talu_set_response_format` because:
/// - No schema parsing or validation (already done at compile time)
/// Build GrammarConfig from C config struct.
fn buildGrammarConfig(config: ?*const ValidateConfigC) GrammarConfig {
    const cfg = config orelse return GrammarConfig{};
    return .{
        .allow_thinking = cfg.allow_thinking,
        .max_thinking_tokens = cfg.max_thinking_tokens,
        .start_marker = if (cfg.start_marker) |m| std.mem.span(m) else null,
        .soft_limit_ratio = cfg.soft_limit_ratio,
        .soft_limit_bias = cfg.soft_limit_bias,
    };
}

/// - O(1) cache lookup using the stored key
///
/// Parameters:
///   chat_handle: Chat session handle
///   grammar_handle: Pre-compiled grammar handle from talu_grammar_compile
///   config: Optional configuration (thinking mode, soft limits)
///   stop_tokens: EOS token IDs
///   stop_tokens_len: Number of stop tokens
///   prefix_token_ids: Optional prefix tokens to prepend
///   prefix_token_ids_len: Number of prefix tokens
///
/// Returns 0 on success, negative error code on failure.
pub export fn talu_set_response_format_handle(
    chat_handle: ?*ChatHandle,
    grammar_handle: ?*GrammarHandle,
    config: ?*const ValidateConfigC,
    stop_tokens: ?[*]const u32,
    stop_tokens_len: usize,
    prefix_token_ids: ?[*]const u32,
    prefix_token_ids_len: usize,
) callconv(.c) i32 {
    capi_error.clearError();

    const chat_state: *Chat = @ptrCast(@alignCast(chat_handle orelse {
        capi_error.setError(error.InvalidHandle, "Invalid chat handle", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));
    const grammar_data = (grammar_handle orelse {
        capi_error.setError(error.InvalidHandle, "Invalid grammar handle", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }).toPtr();

    const stop_slice = if (stop_tokens) |ptr| ptr[0..stop_tokens_len] else &[_]u32{};
    const prefix_slice = if (prefix_token_ids) |ptr| ptr[0..prefix_token_ids_len] else &[_]u32{};

    setResponseFormatSlice(chat_state, grammar_data.schema_json, buildGrammarConfig(config), stop_slice, prefix_slice) catch |err| {
        if (err == error.AllocFailed) {
            capi_error.setError(error.OutOfMemory, "Failed to allocate sampler", .{});
            return @intFromEnum(error_codes.ErrorCode.out_of_memory);
        }
        capi_error.setError(err, "Failed to set response format", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

// ============================================================================
// Low-Level Engine API - Isolated Grammar Operations
// ============================================================================
//
// This API exposes the grammar engine directly for:
// - Isolated benchmarking without model inference
// - Standalone validation (e.g., template output validation)
// - Streaming JSON validation
// - Schema-aware autocomplete

/// Opaque handle for low-level grammar engine operations.
pub const ValidateEngineHandle = opaque {
    fn fromPtr(ptr: *Validator) *ValidateEngineHandle {
        return @ptrCast(ptr);
    }

    fn toPtr(self: *ValidateEngineHandle) *Validator {
        return @ptrCast(@alignCast(self));
    }
};

/// Creates a grammar engine from a JSON schema string.
///
/// The engine provides low-level byte and token validation without the
/// higher-level sampler features (thinking mode, stop tokens).
///
/// Parameters:
///   schema_json: JSON Schema string defining the output structure
///
/// Returns handle on success, null on schema parse error or OOM.
/// Caller must call talu_validate_engine_destroy() when done.
pub export fn talu_validate_engine_create(
    schema_json: [*:0]const u8,
) callconv(.c) ?*ValidateEngineHandle {
    const allocator = std.heap.c_allocator;
    const validator = allocator.create(Validator) catch {
        capi_error.setError(error.OutOfMemory, "Failed to allocate validator", .{});
        return null;
    };
    validator.* = Validator.init(allocator, std.mem.span(schema_json)) catch |e| {
        capi_error.setError(e, "Failed to parse schema", .{});
        allocator.destroy(validator);
        return null;
    };
    return ValidateEngineHandle.fromPtr(validator);
}

/// Destroys a grammar engine and frees all resources.
///
/// Safe to call with null (no-op).
pub export fn talu_validate_engine_destroy(handle: ?*ValidateEngineHandle) callconv(.c) void {
    const validator = (handle orelse return).toPtr();
    validator.deinit();
    std.heap.c_allocator.destroy(validator);
}

/// Resets the engine to initial state.
///
/// Call this to validate a new input without recreating the engine.
///
/// Returns 0 on success, negative error code on failure.
pub export fn talu_validate_engine_reset(handle: ?*ValidateEngineHandle) callconv(.c) i32 {
    capi_error.clearError();
    const validator = (handle orelse {
        capi_error.setError(error.InvalidHandle, "Invalid engine handle", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }).toPtr();
    validator.reset() catch |e| {
        capi_error.setError(e, "Failed to reset engine", .{});
        return @intFromEnum(error_codes.errorToCode(e));
    };
    return 0;
}

/// Checks if the grammar has reached an accepting state (complete valid JSON).
///
/// Returns true if the input processed so far is valid and complete.
pub export fn talu_validate_engine_is_complete(handle: ?*ValidateEngineHandle) callconv(.c) bool {
    const validator = (handle orelse return false).toPtr();
    return validator.isComplete();
}

/// Gets the current position in input (bytes consumed).
pub export fn talu_validate_engine_get_position(handle: ?*ValidateEngineHandle) callconv(.c) usize {
    const validator = (handle orelse return 0).toPtr();
    return validator.getPosition();
}

// ============================================================================
// Byte-Level Operations (No Tokenizer Required)
// ============================================================================

/// Gets valid next bytes from current state.
///
/// Writes 256 bools to out_valid where index corresponds to byte value.
/// out_valid[b] = true means byte b is valid from current state.
///
/// Returns 0 on success, negative error code on failure.
pub export fn talu_validate_engine_get_valid_bytes(
    handle: ?*ValidateEngineHandle,
    out_valid: ?*[256]bool,
) callconv(.c) i32 {
    capi_error.clearError();
    const validator = (handle orelse {
        capi_error.setError(error.InvalidHandle, "Invalid engine handle", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }).toPtr();
    const valid = out_valid orelse {
        capi_error.setError(error.InvalidArgument, "Output buffer is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    validator.getValidFirstBytes(valid);
    return 0;
}

/// Checks if a byte sequence can be accepted from current state.
///
/// This is a read-only check - does NOT advance the engine state.
/// Returns true if the sequence would be valid.
pub export fn talu_validate_engine_can_accept(
    handle: ?*ValidateEngineHandle,
    bytes: ?[*]const u8,
    len: usize,
) callconv(.c) bool {
    const validator = (handle orelse return false).toPtr();
    const data = (bytes orelse return false)[0..len];
    return validator.canAccept(data) catch false;
}

/// Advances engine state by a byte sequence.
///
/// Returns number of bytes successfully consumed.
/// May be less than len if an invalid byte is encountered.
pub export fn talu_validate_engine_advance(
    handle: ?*ValidateEngineHandle,
    bytes: ?[*]const u8,
    len: usize,
) callconv(.c) usize {
    const validator = (handle orelse return 0).toPtr();
    const data = (bytes orelse return 0)[0..len];
    return validator.advance(data) catch 0;
}

/// Validates complete input: resets, advances through all bytes, checks for completion.
///
/// Convenience function for one-shot validation.
/// Returns 1 if valid and complete, 0 if invalid or incomplete, -1 on error.
pub export fn talu_validate_engine_validate(
    handle: ?*ValidateEngineHandle,
    bytes: ?[*]const u8,
    len: usize,
) callconv(.c) i32 {
    capi_error.clearError();
    const validator = (handle orelse {
        capi_error.setError(error.InvalidHandle, "validate engine handle is null", .{});
        return @intFromEnum(error_codes.errorToCode(error.InvalidHandle));
    }).toPtr();
    const data = (bytes orelse {
        capi_error.setError(error.InvalidArgument, "bytes is null", .{});
        return @intFromEnum(error_codes.errorToCode(error.InvalidArgument));
    })[0..len];
    const is_valid = validator.validate(data) catch |err| {
        capi_error.setError(err, "validation failed", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return if (is_valid) 1 else 0;
}

// ============================================================================
// Token-Level Operations (For LLM Integration)
// ============================================================================

fn setResponseFormat(
    chat_state: *Chat,
    schema_json: [*:0]const u8,
    config: GrammarConfig,
    stop_tokens: []const u32,
    prefix_tokens: []const u32,
) !void {
    return setResponseFormatSlice(
        chat_state,
        std.mem.span(schema_json),
        config,
        stop_tokens,
        prefix_tokens,
    );
}

fn setResponseFormatSlice(
    chat_state: *Chat,
    schema_json: []const u8,
    config: GrammarConfig,
    stop_tokens: []const u32,
    prefix_tokens: []const u32,
) !void {
    var sampler = try ConstrainedSampler.init(
        chat_state.allocator,
        schema_json,
        config,
        stop_tokens,
        null,
        if (prefix_tokens.len > 0) prefix_tokens else null,
    );
    errdefer sampler.deinit();

    const sampler_ptr = chat_state.allocator.create(ConstrainedSampler) catch {
        return error.AllocFailed;
    };
    errdefer chat_state.allocator.destroy(sampler_ptr);

    sampler_ptr.* = sampler;
    chat_state.setGrammar(sampler_ptr);
}

// ============================================================================
// Semantic Validation API - Post-Parse Constraint Checking
// ============================================================================
//
// Grammar validation ensures syntactic correctness during generation.
// Semantic validation checks constraints that grammar cannot express:
// - Float min/max (infinite values cannot be enumerated)
// - Large integer ranges (enumeration would explode grammar)
// - additionalProperties: false (cannot express "no other keys")
//
// Typical usage:
// 1. Grammar validates during generation (talu_validate_* or talu_validate_engine_*)
// 2. After generation completes, semantic validator checks remaining constraints
// 3. If semantic validation fails, use error message for retry/correction

const SemanticValidator = validate_mod.SemanticValidator;

/// Opaque handle for semantic validator.
/// Thread safety: NOT thread-safe. Create one per thread.
pub const SemanticValidatorHandle = opaque {
    fn fromPtr(ptr: *SemanticValidator) *SemanticValidatorHandle {
        return @ptrCast(ptr);
    }

    fn toPtr(self: *SemanticValidatorHandle) *SemanticValidator {
        return @ptrCast(@alignCast(self));
    }
};

/// Result of semantic validation (C-compatible struct).
pub const SemanticViolationC = extern struct {
    /// JSON path to the violating value (e.g., "$.person.age")
    path: ?[*:0]const u8 = null,
    /// Human-readable error message
    message: ?[*:0]const u8 = null,
    /// Constraint type that was violated (0=min, 1=max, 2=excl_min, 3=excl_max, 4=additional_props)
    constraint_type: i32 = 0,
};

/// Creates a semantic validator from a JSON schema string.
///
/// The validator checks constraints that grammar-based validation cannot express:
/// - Number min/max constraints (especially for floats)
/// - additionalProperties: false
///
/// Parameters:
///   schema_json: JSON Schema string
///
/// Returns handle on success, null on schema parse error or OOM.
/// Caller must call talu_semantic_validator_destroy() when done.
pub export fn talu_semantic_validator_create(
    schema_json: [*:0]const u8,
) callconv(.c) ?*SemanticValidatorHandle {
    capi_error.clearError();
    const allocator = std.heap.c_allocator;

    const validator = allocator.create(SemanticValidator) catch {
        capi_error.setError(error.OutOfMemory, "Failed to allocate semantic validator", .{});
        return null;
    };
    errdefer allocator.destroy(validator);

    validator.* = SemanticValidator.init(allocator, std.mem.span(schema_json)) catch |err| {
        capi_error.setError(err, "Failed to parse schema for semantic validation", .{});
        return null;
    };

    return SemanticValidatorHandle.fromPtr(validator);
}

/// Destroys a semantic validator and frees all resources.
///
/// Safe to call with null (no-op).
pub export fn talu_semantic_validator_destroy(handle: ?*SemanticValidatorHandle) callconv(.c) void {
    const validator = (handle orelse return).toPtr();
    validator.deinit();
    std.heap.c_allocator.destroy(validator);
}

/// Validates JSON string against semantic constraints.
///
/// Checks constraints that grammar cannot express (number ranges, additionalProperties).
/// Should be called after grammar validation passes.
///
/// Convert constraint type to C integer.
fn constraintTypeToInt(ct: anytype) u8 {
    return switch (ct) {
        .number_minimum => 0,
        .number_maximum => 1,
        .number_exclusive_minimum => 2,
        .number_exclusive_maximum => 3,
        .additional_properties => 4,
        .type_mismatch => 5,
        .required_property => 6,
    };
}

/// Parameters:
///   handle: Semantic validator handle
///   json_str: The JSON string to validate
///   json_len: Length of the JSON string
///   out_violation: Output struct populated if validation fails
///
/// Returns:
///   0 = valid (no violation)
///   1 = violation found (check out_violation for details)
///   negative = error (check talu_get_last_error())
pub export fn talu_semantic_validator_check(
    handle: ?*SemanticValidatorHandle,
    json_str: ?[*]const u8,
    json_len: usize,
    out_violation: ?*SemanticViolationC,
) callconv(.c) i32 {
    capi_error.clearError();

    const validator = (handle orelse {
        capi_error.setError(error.InvalidHandle, "Invalid semantic validator handle", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }).toPtr();
    const json_data = (json_str orelse {
        capi_error.setError(error.InvalidArgument, "JSON string is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    })[0..json_len];
    const violation_out = out_violation orelse {
        capi_error.setError(error.InvalidArgument, "Output violation struct is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    violation_out.* = .{};

    const result = validator.validate(json_data) catch |err| {
        capi_error.setError(err, "Semantic validation failed", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    if (result) |v| {
        violation_out.path = @ptrCast(v.path.ptr);
        violation_out.message = @ptrCast(v.message.ptr);
        violation_out.constraint_type = constraintTypeToInt(v.constraint_type);
        return 1;
    }
    return 0;
}

// =============================================================================
// Fuzz Tests
// =============================================================================

test "fuzz talu_grammar_compile" {
    try std.testing.fuzz({}, struct {
        fn testOne(_: void, input: []const u8) !void {
            const alloc = std.testing.allocator;
            const json_z = try alloc.dupeZ(u8, input);
            defer alloc.free(json_z);
            if (talu_grammar_compile(json_z.ptr)) |handle| {
                talu_grammar_free(handle);
            }
        }
    }.testOne, .{});
}

test "fuzz talu_json_schema_compile" {
    // Alias: the C API entry point is talu_grammar_compile.
    try std.testing.fuzz({}, struct {
        fn testOne(_: void, input: []const u8) !void {
            const alloc = std.testing.allocator;
            const json_z = try alloc.dupeZ(u8, input);
            defer alloc.free(json_z);
            if (talu_grammar_compile(json_z.ptr)) |handle| {
                talu_grammar_free(handle);
            }
        }
    }.testOne, .{});
}

test "fuzz talu_semantic_validator_create" {
    try std.testing.fuzz({}, struct {
        fn testOne(_: void, input: []const u8) !void {
            const alloc = std.testing.allocator;
            const schema_z = try alloc.dupeZ(u8, input);
            defer alloc.free(schema_z);
            if (talu_semantic_validator_create(schema_z.ptr)) |handle| {
                talu_semantic_validator_destroy(handle);
            }
        }
    }.testOne, .{});
}

test "fuzz talu_semantic_validator_check" {
    const schema = "{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"number\"}}}";
    try std.testing.fuzz({}, struct {
        fn testOne(_: void, input: []const u8) !void {
            const validator = talu_semantic_validator_create(schema) orelse return;
            defer talu_semantic_validator_destroy(validator);

            var violation: SemanticViolationC = .{};
            const rc = talu_semantic_validator_check(validator, input.ptr, input.len, &violation);
            if (rc == 0) {
                try std.testing.expect(violation.path == null);
                try std.testing.expect(violation.message == null);
            }
        }
    }.testOne, .{});
}

test "talu_grammar_compile invalid schema maps to invalid_argument" {
    capi_error.clearError();
    const handle = talu_grammar_compile("{");
    try std.testing.expect(handle == null);
    try std.testing.expectEqual(@as(i32, @intFromEnum(error_codes.ErrorCode.invalid_argument)), capi_error.talu_last_error_code());
}
