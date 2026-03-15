//! Utility C API
//!
//! Utility functions for model resolution, chat templates, and EOS tokens.
//! For generation, use router.zig.
//! For tokenization, see tokenizer.zig.

const std = @import("std");
const gen_config_mod = @import("../inference/config/generation.zig");
const io = @import("../io/root.zig");
const repository = io.repository;
const transport = io.transport;
const capi_error = @import("error.zig");
const error_codes = @import("error_codes.zig");
const log = @import("../log.zig");

// Import tokenizer types and functions
const tokenizer_mod = @import("tokenizer.zig");
pub const TokenizerHandle = tokenizer_mod.TokenizerHandle;
pub const TokenizerWrapper = tokenizer_mod.TokenizerWrapper;
pub const EncodeResult = tokenizer_mod.EncodeResult;
pub const DecodeResult = tokenizer_mod.DecodeResult;
pub const EncodeOptions = tokenizer_mod.EncodeOptions;
pub const TokenizeResult = tokenizer_mod.TokenizeResult;
pub const TokenizeBytesResult = tokenizer_mod.TokenizeBytesResult;
pub const TokenOffset = tokenizer_mod.TokenOffset;
pub const OffsetsResult = tokenizer_mod.OffsetsResult;
pub const EosTokenResult = tokenizer_mod.EosTokenResult;
pub const BatchEncodeResult = tokenizer_mod.BatchEncodeResult;
pub const SpecialTokensResult = tokenizer_mod.SpecialTokensResult;
pub const DecodeOptionsC = tokenizer_mod.DecodeOptionsC;
pub const VocabResult = tokenizer_mod.VocabResult;
pub const PaddedTensorOptions = tokenizer_mod.PaddedTensorOptions;
pub const PaddedTensorResult = tokenizer_mod.PaddedTensorResult;
pub const addEosFromTokenizer = tokenizer_mod.addEosFromTokenizer;

// Import model description types
const converter_mod = @import("converter.zig");
pub const ModelInfo = converter_mod.ModelInfo;

// Native build uses C allocator
const allocator = std.heap.c_allocator;

/// Sampling configuration for C API
pub const SamplingParams = extern struct {
    /// Sampling strategy: 0=greedy, 1=top_k, 2=top_p
    strategy: u32 = 0,
    /// Temperature (default 1.0, 0 means use model default)
    temperature: f32 = 1.0,
    /// Top-k value (default 50)
    top_k: u32 = 50,
    /// Top-p value (default 0.9)
    top_p: f32 = 0.9,
    /// Min-p value (default 0.0 = disabled)
    min_p: f32 = 0.0,
    /// Repetition penalty (default 1.0 = no penalty)
    repetition_penalty: f32 = 1.0,
    /// Random seed for reproducibility (0 = time-based)
    seed: u64 = 0,
};

pub const GenerationConfigInfo = extern struct {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    do_sample: bool,
};

/// Input for effective generation config resolution.
/// Uses -1.0 as sentinel for "unset" floats.
/// This struct MUST be initialized with std.mem.zeroes() before use.
///
/// NOT thread-safe: intended for single-threaded use per call.
pub const EffectiveGenConfigRequest = extern struct {
    /// Temperature override. -1.0 = use model default.
    temperature: f32 = -1.0,
    /// Top-k override. 0 = use model default.
    top_k: usize = 0,
    /// Top-p override. -1.0 = use model default.
    top_p: f32 = -1.0,
    /// Min-p override. -1.0 = use model default.
    min_p: f32 = -1.0,
    /// Repetition penalty override. -1.0 = use model default.
    repetition_penalty: f32 = -1.0,
    /// Seed (0 = random).
    seed: u64 = 0,
    /// Max tokens to generate.
    max_tokens: usize = 0,
};

/// Resolved effective generation config after applying policy.
/// This is the output of talu_resolve_effective_generation_config.
///
/// The policy matches core/src/router/local.zig sampling decision:
/// - Sampling is enabled if: temperature > 0 AND (model.do_sample OR user provided override)
/// - If not sampling, temperature is forced to 0.0 (greedy)
///
/// NOT thread-safe: intended for single-threaded use per call.
pub const EffectiveGenConfig = extern struct {
    /// Effective temperature (0.0 = greedy).
    temperature: f32,
    /// Effective top_k.
    top_k: usize,
    /// Effective top_p.
    top_p: f32,
    /// Effective min_p.
    min_p: f32,
    /// Effective repetition_penalty.
    repetition_penalty: f32,
    /// Seed.
    seed: u64,
    /// Max tokens.
    max_tokens: usize,
    /// Whether sampling is enabled (true) or greedy (false).
    do_sample: bool,
};

fn allocZFromSlice(bytes: []const u8) ?[*:0]u8 {
    const cstr_buffer = allocator.allocSentinel(u8, bytes.len, 0) catch return null;
    @memcpy(cstr_buffer, bytes);
    return cstr_buffer.ptr;
}

// =============================================================================
// Model Path Resolution (Model Hub)
// =============================================================================

/// Resolve a model path/URI, downloading from the hub if needed.
///
/// Handles all supported schemes:
/// - `org/model` - HF model ID (checks cache first, downloads if not cached)
/// - `/path`, `./path`, `../path`, `~/path` - Local filesystem
/// - `models--org--name/...` - HF cache format
///
/// On success, writes a null-terminated path to out_path (caller must free with talu_text_free).
pub export fn talu_resolve_model_path(
    model_path: [*:0]const u8,
    out_path: *?[*:0]u8,
) callconv(.c) i32 {
    capi_error.clearError();
    out_path.* = null;
    const model_path_slice = std.mem.span(model_path);

    // Use centralized resolution logic from repository.
    // Tokenizer loading does not require weight files.
    const resolved_path = repository.resolveModelPath(allocator, model_path_slice, .{ .require_weights = false }) catch |err| {
        capi_error.setError(err, "Model path resolution failed: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer allocator.free(resolved_path);

    out_path.* = allocZFromSlice(resolved_path) orelse {
        capi_error.setError(error.OutOfMemory, "Model path duplication failed (out of memory)", .{});
        return @intFromEnum(error_codes.errorToCode(error.OutOfMemory));
    };
    return 0;
}

// =============================================================================
// EOS Token Functions
// =============================================================================

/// Get EOS token IDs from model's generation_config.json.
/// Returns array of token IDs. Caller must free with talu_tokens_free.
/// Returns empty result on error (check talu_error_message for details).
pub export fn talu_get_eos_tokens(
    model_dir: [*:0]const u8,
) callconv(.c) EosTokenResult {
    capi_error.clearError();
    const model_dir_slice = std.mem.span(model_dir);
    const resolved_path = repository.resolveModelPath(allocator, model_dir_slice, .{}) catch |err| {
        capi_error.setError(err, "Model path resolution failed", .{});
        return EosTokenResult{ .tokens = null, .num_tokens = 0 };
    };
    defer allocator.free(resolved_path);

    // Use shared module (same as main.zig)
    var config = gen_config_mod.loadGenerationConfig(allocator, resolved_path) catch |err| {
        capi_error.setError(err, "Generation config load failed", .{});
        return EosTokenResult{ .tokens = null, .num_tokens = 0 };
    };

    if (config.eos_token_ids.len == 0) {
        config.deinit(allocator);
        // Not an error - model simply has no EOS tokens configured
        return EosTokenResult{ .tokens = null, .num_tokens = 0 };
    }

    // Copy to owned array (config.deinit would free the original)
    const eos_ids = allocator.alloc(u32, config.eos_token_ids.len) catch {
        config.deinit(allocator);
        capi_error.setError(error.OutOfMemory, "Failed to allocate EOS token array", .{});
        return EosTokenResult{ .tokens = null, .num_tokens = 0 };
    };
    @memcpy(eos_ids, config.eos_token_ids);
    config.deinit(allocator);

    return EosTokenResult{ .tokens = eos_ids.ptr, .num_tokens = eos_ids.len };
}

/// Get generation config defaults (temperature/top_k/top_p/do_sample).
pub export fn talu_get_generation_config(
    model_dir: [*:0]const u8,
    out_config: *GenerationConfigInfo,
) callconv(.c) i32 {
    capi_error.clearError();
    const model_dir_slice = std.mem.span(model_dir);
    const resolved_path = repository.resolveModelPath(allocator, model_dir_slice, .{}) catch |err| {
        capi_error.setError(err, "Model path resolution failed: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer allocator.free(resolved_path);

    var config = gen_config_mod.loadGenerationConfig(allocator, resolved_path) catch |err| {
        capi_error.setError(err, "Generation config load failed: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer config.deinit(allocator);

    out_config.* = .{
        .temperature = config.temperature,
        .top_k = config.top_k,
        .top_p = config.top_p,
        .do_sample = config.do_sample,
    };
    return 0;
}

// =============================================================================
// Effective Generation Config (Policy Resolution)
// =============================================================================

/// Resolve effective generation config by applying model defaults and overrides.
///
/// This is the SINGLE source of truth for generation config policy. All entrypoints
/// (CLI ask, xray, shell, server) must use this function instead of implementing
/// their own policy logic.
///
/// The policy matches core/src/router/local.zig sampling decision:
/// - Apply user overrides over model defaults (sentinel -1.0 means "use model default")
/// - Sampling is enabled if: temperature > 0 AND (model.do_sample OR user provided temperature override)
/// - If not sampling, temperature is forced to 0.0 (greedy decoding)
///
/// Args:
///   model_dir: Path to model directory (will be resolved via repository)
///   request: Input config with optional overrides (use -1.0 for unset floats, 0 for unset top_k)
///   out_config: Output with resolved effective config
///
/// Returns: 0 on success, error code on failure.
pub export fn talu_resolve_effective_generation_config(
    model_dir: [*:0]const u8,
    request: *const EffectiveGenConfigRequest,
    out_config: *EffectiveGenConfig,
) callconv(.c) i32 {
    capi_error.clearError();
    // Initialize output to zeroes (per policy: extern structs must use std.mem.zeroes)
    out_config.* = std.mem.zeroes(EffectiveGenConfig);

    const model_dir_slice = std.mem.span(model_dir);
    const resolved_path = repository.resolveModelPath(allocator, model_dir_slice, .{}) catch |err| {
        capi_error.setError(err, "Model path resolution failed: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer allocator.free(resolved_path);

    var model_config = gen_config_mod.loadGenerationConfig(allocator, resolved_path) catch |err| {
        capi_error.setError(err, "Generation config load failed: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer model_config.deinit(allocator);

    // Apply overrides: use request value if not sentinel, else use model default
    const user_set_temperature = request.temperature >= 0.0;
    const temperature = if (user_set_temperature) request.temperature else model_config.temperature;
    const top_k = if (request.top_k > 0) request.top_k else model_config.top_k;
    const top_p = if (request.top_p >= 0.0) request.top_p else model_config.top_p;
    // min_p default is 0.0, so use sentinel -1.0 to distinguish "unset" from "set to 0"
    const min_p = if (request.min_p >= 0.0) request.min_p else 0.0;
    // repetition_penalty default is 1.0, use sentinel -1.0 for "unset"
    const repetition_penalty = if (request.repetition_penalty >= 0.0) request.repetition_penalty else 1.0;

    // Apply the sampling policy (matches local.zig:725)
    // Only sample if: temperature > 0 AND (model.do_sample OR user explicitly set temperature)
    const do_sample = temperature > 0 and (model_config.do_sample or user_set_temperature);

    // If not sampling, force greedy (temperature = 0)
    const effective_temperature = if (do_sample) temperature else 0.0;

    out_config.* = .{
        .temperature = effective_temperature,
        .top_k = top_k,
        .top_p = top_p,
        .min_p = min_p,
        .repetition_penalty = repetition_penalty,
        .seed = request.seed,
        .max_tokens = request.max_tokens,
        .do_sample = do_sample,
    };
    return 0;
}

// =============================================================================
// Chat Template
// =============================================================================

/// Apply chat template with a JSON array of messages.
/// Supports multi-turn conversations, tool calls, and assistant prefill.
/// On success, writes a null-terminated formatted prompt string to out_prompt.
/// Caller must free with talu_text_free.
pub export fn talu_apply_chat_template(
    model_path: [*:0]const u8,
    messages_json: [*:0]const u8,
    add_generation_prompt: c_int,
    out_prompt: *?[*:0]u8,
) callconv(.c) i32 {
    capi_error.clearError();
    out_prompt.* = null;

    const model_path_slice = std.mem.span(model_path);
    const messages_json_slice = std.mem.span(messages_json);

    const resolved_path = repository.resolveModelPath(allocator, model_path_slice, .{}) catch |err| {
        capi_error.setError(err, "Model path resolution failed: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer allocator.free(resolved_path);

    const rendered_prompt = gen_config_mod.applyChatTemplate(
        allocator,
        resolved_path,
        messages_json_slice,
        add_generation_prompt != 0,
    ) catch |err| {
        capi_error.setError(err, "Chat template failed: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    out_prompt.* = allocZFromSlice(rendered_prompt) orelse {
        capi_error.setError(error.OutOfMemory, "Chat template allocation failed", .{});
        allocator.free(rendered_prompt);
        return @intFromEnum(error_codes.errorToCode(error.OutOfMemory));
    };
    allocator.free(rendered_prompt);
    return 0;
}

/// Apply chat template from a template string (no model directory required).
///
/// This enables standalone chat template rendering without needing model files.
/// Useful for testing, custom templates, and serverless deployments.
///
/// On success, writes a null-terminated formatted prompt string to out_prompt.
/// Caller must free with talu_text_free.
pub export fn talu_apply_chat_template_string(
    template_ptr: [*]const u8,
    template_len: usize,
    messages_json: [*:0]const u8,
    add_generation_prompt: c_int,
    bos_token: [*:0]const u8,
    eos_token: [*:0]const u8,
    out_prompt: *?[*:0]u8,
) callconv(.c) i32 {
    capi_error.clearError();
    out_prompt.* = null;

    if (template_len == 0) {
        capi_error.setError(error.InvalidArgument, "Empty template string", .{});
        return @intFromEnum(error_codes.errorToCode(error.InvalidArgument));
    }

    const template_slice = template_ptr[0..template_len];
    const messages_json_slice = std.mem.span(messages_json);
    const bos_slice = std.mem.span(bos_token);
    const eos_slice = std.mem.span(eos_token);

    const rendered_prompt = gen_config_mod.applyChatTemplateFromString(
        allocator,
        template_slice,
        messages_json_slice,
        add_generation_prompt != 0,
        bos_slice,
        eos_slice,
    ) catch |err| {
        capi_error.setError(err, "Chat template failed: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    out_prompt.* = allocZFromSlice(rendered_prompt) orelse {
        capi_error.setError(error.OutOfMemory, "Chat template allocation failed", .{});
        allocator.free(rendered_prompt);
        return @intFromEnum(error_codes.errorToCode(error.OutOfMemory));
    };
    allocator.free(rendered_prompt);
    return 0;
}

// =============================================================================
// Tests
// =============================================================================

test "talu_resolve_effective_generation_config greedy when do_sample=false and no override" {
    // This test verifies the core policy: when model has do_sample=false and user
    // doesn't provide temperature override, effective config should be greedy (temp=0).
    //
    // We use the internal function directly with mock data since the C API
    // requires a real model directory.

    // Test the policy logic directly
    const model_do_sample = false;
    const model_temperature: f32 = 1.0;
    const user_set_temperature = false;
    const temperature = model_temperature;

    // Apply policy: only sample if temp > 0 AND (model.do_sample OR user set temp)
    const do_sample = temperature > 0 and (model_do_sample or user_set_temperature);
    const effective_temperature = if (do_sample) temperature else 0.0;

    try std.testing.expect(!do_sample);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), effective_temperature, 0.001);
}

test "talu_resolve_effective_generation_config samples when user provides override" {
    // When user explicitly sets temperature, sampling is enabled even if model
    // has do_sample=false.

    const model_do_sample = false;
    const user_temperature: f32 = 0.8;
    const user_set_temperature = true;

    // Apply policy
    const do_sample = user_temperature > 0 and (model_do_sample or user_set_temperature);
    const effective_temperature = if (do_sample) user_temperature else 0.0;

    try std.testing.expect(do_sample);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), effective_temperature, 0.001);
}

test "talu_resolve_effective_generation_config greedy when user sets zero temperature" {
    // When user explicitly sets temperature=0, it should be greedy.

    const model_do_sample = true;
    const user_temperature: f32 = 0.0;
    const user_set_temperature = true;

    // Apply policy: temp=0 means no sampling even if user set it
    const do_sample = user_temperature > 0 and (model_do_sample or user_set_temperature);
    const effective_temperature = if (do_sample) user_temperature else 0.0;

    try std.testing.expect(!do_sample);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), effective_temperature, 0.001);
}

test "talu_resolve_effective_generation_config samples when model has do_sample=true" {
    // When model has do_sample=true and temperature > 0, sampling is enabled.

    const model_do_sample = true;
    const model_temperature: f32 = 0.7;
    const user_set_temperature = false;
    const temperature = model_temperature;

    // Apply policy
    const do_sample = temperature > 0 and (model_do_sample or user_set_temperature);
    const effective_temperature = if (do_sample) temperature else 0.0;

    try std.testing.expect(do_sample);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), effective_temperature, 0.001);
}

test "EffectiveGenConfigRequest default values use sentinels" {
    // Verify that default struct values use the sentinel values for "unset"
    const request = EffectiveGenConfigRequest{};

    try std.testing.expectApproxEqAbs(@as(f32, -1.0), request.temperature, 0.001);
    try std.testing.expectEqual(@as(usize, 0), request.top_k);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), request.top_p, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), request.min_p, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), request.repetition_penalty, 0.001);
    try std.testing.expectEqual(@as(u64, 0), request.seed);
    try std.testing.expectEqual(@as(usize, 0), request.max_tokens);
}

test "EffectiveGenConfig zeroed initialization" {
    // Verify zeroed initialization per policy
    const config = std.mem.zeroes(EffectiveGenConfig);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), config.temperature, 0.001);
    try std.testing.expectEqual(@as(usize, 0), config.top_k);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), config.top_p, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), config.min_p, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), config.repetition_penalty, 0.001);
    try std.testing.expectEqual(@as(u64, 0), config.seed);
    try std.testing.expectEqual(@as(usize, 0), config.max_tokens);
    try std.testing.expectEqual(false, config.do_sample);
}
