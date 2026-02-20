//! C API Bridge for Router
//!
//! Provides high-level generation functions that bridge between C-compatible
//! types and the Zig router/engine types. This keeps all conversion logic
//! out of the CAPI layer, which only handles external contract definitions.
//!
//! The CAPI calls these functions directly - no logic in capi/router.zig.

const std = @import("std");
const local = @import("local.zig");
const root = @import("root.zig");
const spec = @import("spec.zig");
const http_engine_mod = @import("http_engine.zig");
const iterator_mod = @import("iterator.zig");
const commit_mod = @import("commit.zig");
const responses_mod = @import("../responses/root.zig");
const sampler = @import("../inference/root.zig").sampling;
const tokenizer_mod = @import("../tokenizer/root.zig");
const error_codes = @import("../capi/error_codes.zig");
const capi_error = @import("../capi/error.zig");
const log = @import("../log.zig");

const LocalEngine = local.LocalEngine;
const HttpEngine = http_engine_mod.HttpEngine;
const InferenceBackend = spec.InferenceBackend;
const GenerateOptions = local.GenerateOptions;
const Chat = responses_mod.Chat;
const ContentType = responses_mod.ContentType;

// =============================================================================
// C-Compatible Input Types
// =============================================================================

/// Content part input for generation.
/// This is a simplified input-only struct for generation requests.
/// See capi/messages.zig CContentPart for the full read struct with metadata.
pub const GenerateContentPart = extern struct {
    content_type: u8,
    data_ptr: [*]const u8,
    data_len: usize,
    mime_ptr: ?[*:0]const u8,
};

fn contentTypeFromGeneratePart(raw: u8) ContentType {
    return switch (raw) {
        0 => .input_text,
        1 => .input_image,
        2 => .input_audio,
        3 => .input_video,
        4 => .input_file,
        else => .unknown,
    };
}

fn appendUserMessageFromParts(chat: *Chat, parts: []const GenerateContentPart) !void {
    log.trace("router", "Appending user message", .{ .parts = parts.len }, @src());
    const msg = try chat.conv.appendEmptyMessage(.user);
    for (parts) |c_part| {
        const content_type = contentTypeFromGeneratePart(c_part.content_type);
        const part = try chat.conv.addContentPart(msg, content_type);
        try part.appendData(chat.conv.allocator, c_part.data_ptr[0..c_part.data_len]);
    }
    chat.conv.finalizeItem(msg);
}

/// Logit bias entry from C API.
pub const CLogitBiasEntry = extern struct {
    token_id: u32,
    bias: f32,
};

/// Generation configuration from C API.
pub const CGenerateConfig = extern struct {
    max_tokens: usize = 0,
    temperature: f32 = -1.0,
    top_k: usize = 0,
    top_p: f32 = -1.0,
    min_p: f32 = -1.0,
    repetition_penalty: f32 = 0.0,
    stop_sequences: ?[*]const [*:0]const u8 = null,
    stop_sequence_count: usize = 0,
    logit_bias: ?[*]const CLogitBiasEntry = null,
    logit_bias_count: usize = 0,
    seed: u64 = 0,
    template_override: ?[*:0]const u8 = null,
    extra_context_json: ?[*:0]const u8 = null,
    /// Tool definitions as JSON array. Format: [{"type":"function","function":{...}}]
    tools_json: ?[*:0]const u8 = null,
    /// Tool choice strategy: "auto", "required", "none", or specific function name.
    tool_choice: ?[*:0]const u8 = null,
    /// Optional stop flag for cancellation. When set to true (non-zero), generation stops.
    /// This allows external cancellation (e.g., client disconnect, asyncio.CancelledError)
    /// without waiting for the next callback invocation.
    /// The pointer must remain valid for the duration of generation.
    stop_flag: ?*const std.atomic.Value(bool) = null,
    /// Extra body fields for remote API requests as a JSON object string.
    /// These are merged into the request body for OpenAI-compatible APIs.
    /// Useful for provider-specific parameters not covered by standard config.
    /// Example: "{\"repetition_penalty\": 1.1, \"top_a\": 0.5}"
    extra_body_json: ?[*:0]const u8 = null,

    /// When non-zero, preserve raw model output bytes without reasoning-tag filtering.
    /// Default (0) keeps current behavior (reasoning tags parsed into typed items).
    raw_output: u8 = 0,

    /// Optional prefill progress callback. Called once per transformer layer
    /// during prefill (not decode). Signature: fn(completed_layers, total_layers, userdata).
    prefill_progress_fn: ?*const fn (usize, usize, ?*anyopaque) callconv(.c) void = null,
    prefill_progress_data: ?*anyopaque = null,
};

// =============================================================================
// Result Types
// =============================================================================

/// C-compatible tool call reference.
pub const CToolCallRef = extern struct {
    /// Index in Conversation.items
    item_index: usize,
    /// Unique identifier for this call (null-terminated).
    call_id: ?[*:0]const u8,
    /// Function name (null-terminated).
    name: ?[*:0]const u8,
    /// Function arguments (null-terminated JSON string).
    arguments: ?[*:0]const u8,
};

/// Finish reason enum values (matches inference/root.zig FinishReason).
pub const CFinishReason = enum(u8) {
    /// Generation stopped due to EOS token.
    eos_token = 0,
    /// Maximum token limit reached.
    length = 1,
    /// A stop sequence was matched.
    stop_sequence = 2,
    /// Model requested tool/function calls.
    tool_calls = 3,
    /// Content was filtered (safety).
    content_filter = 4,
    /// Request was cancelled (e.g., client disconnect, stop flag set).
    cancelled = 5,
};

fn finishReasonToString(reason: CFinishReason) [:0]const u8 {
    return switch (reason) {
        .eos_token => "stop",
        .length => "length",
        .stop_sequence => "stop_sequence",
        .tool_calls => "tool_calls",
        .content_filter => "content_filter",
        .cancelled => "cancelled",
    };
}

fn httpFinishReasonToC(reason: http_engine_mod.FinishReason) CFinishReason {
    return switch (reason) {
        .stop => .eos_token,
        .length => .length,
        .tool_calls => .tool_calls,
        .content_filter => .content_filter,
        .unknown => .eos_token,
    };
}

/// Generation result (internal, Zig slice).
pub const GenerateResult = struct {
    text: ?[]const u8 = null,
    token_count: usize = 0,
    prompt_tokens: usize = 0,
    completion_tokens: usize = 0,
    prefill_ns: u64 = 0,
    generation_ns: u64 = 0,
    error_code: i32 = 0,
    finish_reason: CFinishReason = .eos_token,
    tool_calls: ?[]const local.ToolCallRef = null,

    pub fn deinit(self: *GenerateResult, allocator: std.mem.Allocator) void {
        if (self.text) |t| allocator.free(t);
        // Note: tool_calls are owned by the router result, freed separately
    }
};

/// C-compatible generation result (null-terminated string).
pub const CGenerateResult = extern struct {
    text: ?[*:0]u8,
    token_count: usize,
    prompt_tokens: usize,
    completion_tokens: usize,
    prefill_ns: u64,
    generation_ns: u64,
    error_code: i32,
    /// Why generation stopped (CFinishReason enum value).
    finish_reason: u8 = 0,
    _padding: [3]u8 = .{0} ** 3,
    /// Array of tool calls (if finish_reason == tool_calls).
    tool_calls: ?[*]const CToolCallRef = null,
    /// Number of tool calls in the array.
    tool_call_count: usize = 0,
};

/// Convert GenerateResult to C-compatible result.
/// Copies text to null-terminated C string.
pub fn toCResult(allocator: std.mem.Allocator, result: GenerateResult) CGenerateResult {
    if (result.error_code != 0) {
        var ret = std.mem.zeroes(CGenerateResult);
        ret.error_code = result.error_code;
        return ret;
    }

    const text = result.text orelse {
        var ret = std.mem.zeroes(CGenerateResult);
        ret.error_code = @intFromEnum(error_codes.ErrorCode.internal_error);
        return ret;
    };

    const cstr = allocator.allocSentinel(u8, text.len, 0) catch {
        var ret = std.mem.zeroes(CGenerateResult);
        ret.token_count = result.token_count;
        ret.prompt_tokens = result.prompt_tokens;
        ret.completion_tokens = result.completion_tokens;
        ret.prefill_ns = result.prefill_ns;
        ret.generation_ns = result.generation_ns;
        ret.error_code = @intFromEnum(error_codes.ErrorCode.out_of_memory);
        return ret;
    };
    @memcpy(cstr, text);
    allocator.free(text);

    // Convert tool calls to C format if present
    var c_tool_calls: ?[*]const CToolCallRef = null;
    var tool_call_count: usize = 0;
    if (result.tool_calls) |calls| {
        tool_call_count = calls.len;
        if (calls.len > 0) {
            const c_calls = allocator.alloc(CToolCallRef, calls.len) catch {
                // On OOM, just skip tool calls
                var ret = std.mem.zeroes(CGenerateResult);
                ret.text = cstr.ptr;
                ret.token_count = result.token_count;
                ret.prompt_tokens = result.prompt_tokens;
                ret.completion_tokens = result.completion_tokens;
                ret.prefill_ns = result.prefill_ns;
                ret.generation_ns = result.generation_ns;
                ret.finish_reason = @intFromEnum(result.finish_reason);
                return ret;
            };
            for (calls, 0..) |call, i| {
                var c_call = std.mem.zeroes(CToolCallRef);
                c_call.item_index = call.item_index;
                c_call.call_id = allocator.dupeZ(u8, call.call_id) catch null;
                c_call.name = allocator.dupeZ(u8, call.name) catch null;
                c_call.arguments = allocator.dupeZ(u8, call.arguments) catch null;
                c_calls[i] = c_call;
            }
            c_tool_calls = c_calls.ptr;
        }
    }

    var ret = std.mem.zeroes(CGenerateResult);
    ret.text = cstr.ptr;
    ret.token_count = result.token_count;
    ret.prompt_tokens = result.prompt_tokens;
    ret.completion_tokens = result.completion_tokens;
    ret.prefill_ns = result.prefill_ns;
    ret.generation_ns = result.generation_ns;
    ret.finish_reason = @intFromEnum(result.finish_reason);
    ret.tool_calls = c_tool_calls;
    ret.tool_call_count = tool_call_count;
    return ret;
}

/// Free a C generation result.
pub fn freeResult(allocator: std.mem.Allocator, result: *CGenerateResult) void {
    if (result.text) |text| {
        allocator.free(text[0 .. std.mem.len(text) + 1]);
        result.text = null;
    }
    // Free tool calls array and its strings
    if (result.tool_calls) |calls| {
        for (0..result.tool_call_count) |i| {
            const call = calls[i];
            if (call.call_id) |cid| {
                allocator.free(cid[0 .. std.mem.len(cid) + 1]);
            }
            if (call.name) |n| {
                allocator.free(n[0 .. std.mem.len(n) + 1]);
            }
            if (call.arguments) |args| {
                allocator.free(args[0 .. std.mem.len(args) + 1]);
            }
        }
        const slice: []const CToolCallRef = calls[0..result.tool_call_count];
        allocator.free(slice);
        result.tool_calls = null;
        result.tool_call_count = 0;
    }
}

/// Free internal ToolCallRef slice (used for ownership transfer error paths).
fn freeToolCallRefs(allocator: std.mem.Allocator, tool_calls: ?[]const local.ToolCallRef) void {
    const calls = tool_calls orelse return;
    for (calls) |call| {
        allocator.free(call.call_id);
        allocator.free(call.name);
        allocator.free(call.arguments);
    }
    allocator.free(calls);
}

fn freeToolCallRefsPartial(
    allocator: std.mem.Allocator,
    tool_calls: []const local.ToolCallRef,
    count: usize,
) void {
    const bounded = tool_calls[0..count];
    for (bounded) |call| {
        allocator.free(call.call_id);
        allocator.free(call.name);
        allocator.free(call.arguments);
    }
    allocator.free(tool_calls);
}

// =============================================================================
// Generation API
// =============================================================================

/// Generate a response via the router.
///
/// This is the main entry point for C API generation. It:
/// 1. Validates inputs
/// 2. Resolves and gets/creates engine
/// 3. Creates user message from content parts
/// 4. Builds generation options from C config
/// 5. Runs generation
/// 6. Returns result with owned text
pub fn generate(
    allocator: std.mem.Allocator,
    chat: *Chat,
    parts: []const GenerateContentPart,
    model_id: []const u8,
    resolution: root.ResolutionConfig,
    config: ?*const CGenerateConfig,
) GenerateResult {
    // Validate inputs
    if (parts.len == 0) return .{ .error_code = @intFromEnum(error_codes.ErrorCode.generation_empty_prompt) };

    // Resolve model for routing
    const resolved_id = root.resolveForRouting(model_id) catch return .{ .error_code = @intFromEnum(error_codes.ErrorCode.internal_error) };

    // Get or create engine
    const engine = root.getOrCreateEngineWithConfig(allocator, resolved_id, resolution) catch return .{ .error_code = @intFromEnum(error_codes.ErrorCode.model_not_found) };

    // Create user message from content parts
    appendUserMessageFromParts(chat, parts) catch {
        return .{ .error_code = @intFromEnum(error_codes.ErrorCode.out_of_memory) };
    };

    // Build options
    var built = buildOptions(allocator, config, engine) catch return .{ .error_code = @intFromEnum(error_codes.ErrorCode.out_of_memory) };
    defer built.deinit(allocator);

    // Generate
    const result = engine.generate(chat, built.options) catch return .{ .error_code = @intFromEnum(error_codes.ErrorCode.generation_failed) };
    defer result.deinit(allocator);

    // Copy result text
    const text = allocator.dupe(u8, result.text) catch {
        return .{
            .text = null,
            .token_count = result.generated_tokens,
            .prompt_tokens = result.prompt_tokens,
            .completion_tokens = result.generated_tokens,
            .prefill_ns = result.prefill_ns,
            .generation_ns = result.decode_ns,
            .error_code = @intFromEnum(error_codes.ErrorCode.out_of_memory),
        };
    };

    // Convert finish reason
    const finish_reason: CFinishReason = @enumFromInt(@intFromEnum(result.finish_reason));

    return .{
        .text = text,
        .token_count = result.generated_tokens,
        .prompt_tokens = result.prompt_tokens,
        .completion_tokens = result.generated_tokens,
        .prefill_ns = result.prefill_ns,
        .generation_ns = result.decode_ns,
        .error_code = 0,
        .finish_reason = finish_reason,
        .tool_calls = result.tool_calls,
    };
}

/// Generate using a spec-based InferenceBackend.
///
/// Like generate() but takes an InferenceBackend created from a CanonicalSpec
/// instead of a model ID string. This is the new spec-based API that will replace
/// the string-based API.
///
/// Dispatches to either:
///   - LocalEngine for native inference
///   - HttpEngine for OpenAI-compatible remote inference
pub fn generateWithBackend(
    allocator: std.mem.Allocator,
    chat: *Chat,
    parts: []const GenerateContentPart,
    backend: *InferenceBackend,
    config: ?*const CGenerateConfig,
) GenerateResult {
    log.debug("router", "generateWithBackend", .{
        .parts = parts.len,
        .conv_items = chat.conv.len(),
        .backend = @as(u8, switch (backend.backend) {
            .Local => 0,
            .OpenAICompatible => 1,
            .Unspecified => 2,
        }),
    }, @src());

    // If parts are provided, append a new user message.
    // If parts are empty, continue from the current conversation state
    // (used by agent loops after appending tool call outputs).
    if (parts.len > 0) {
        appendUserMessageFromParts(chat, parts) catch {
            return .{ .error_code = @intFromEnum(error_codes.ErrorCode.out_of_memory) };
        };
    } else if (chat.conv.len() == 0) {
        // Empty conversation with no content — this is an error
        return .{ .error_code = @intFromEnum(error_codes.ErrorCode.generation_empty_prompt) };
    }

    // Dispatch based on backend type
    switch (backend.backend) {
        .Local => |local_engine| {
            return generateWithLocalEngine(allocator, chat, local_engine, config);
        },
        .OpenAICompatible => |http_engine| {
            return generateWithHttpEngine(allocator, chat, http_engine, config);
        },
        .Unspecified => {
            return .{ .error_code = @intFromEnum(error_codes.ErrorCode.invalid_argument) };
        },
    }
}

/// Generate using LocalEngine (native inference).
fn generateWithLocalEngine(
    allocator: std.mem.Allocator,
    chat: *Chat,
    local_engine: *LocalEngine,
    config: ?*const CGenerateConfig,
) GenerateResult {
    // Build options
    var built = buildOptions(allocator, config, local_engine) catch return .{ .error_code = @intFromEnum(error_codes.ErrorCode.out_of_memory) };
    defer built.deinit(allocator);

    log.debug("router", "LocalEngine generate", .{
        .max_tokens = built.options.max_tokens orelse 0,
        .has_tools = @as(u8, @intFromBool(built.options.tools_json != null)),
    }, @src());

    // Generate
    var result = local_engine.generate(chat, built.options) catch |err| {
        if (err == error.UnsupportedContentType)
            capi_error.setError(err, "This model does not support images. Use a vision-language model (e.g. LFM2-VL-450M).", .{})
        else
            capi_error.setError(err, "generate error: {s}", .{@errorName(err)});
        return .{ .error_code = @intFromEnum(error_codes.ErrorCode.generation_failed) };
    };
    log.debug("router", "LocalEngine generate completed", .{
        .prompt_tokens = result.prompt_tokens,
        .completion_tokens = result.generated_tokens,
    }, @src());

    // Transfer tool_calls ownership to the returned GenerateResult before deinit
    // frees them. Null out the field so deinit skips freeing the transferred data.
    const tool_calls = result.tool_calls;
    result.tool_calls = null;
    defer result.deinit(allocator);

    // Copy result text
    const text = allocator.dupe(u8, result.text) catch {
        // Free transferred tool_calls on OOM (we own them now).
        freeToolCallRefs(allocator, tool_calls);
        return .{
            .text = null,
            .token_count = result.generated_tokens,
            .prompt_tokens = result.prompt_tokens,
            .completion_tokens = result.generated_tokens,
            .prefill_ns = result.prefill_ns,
            .generation_ns = result.decode_ns,
            .error_code = @intFromEnum(error_codes.ErrorCode.out_of_memory),
        };
    };

    // Convert finish reason
    const finish_reason: CFinishReason = @enumFromInt(@intFromEnum(result.finish_reason));

    return .{
        .text = text,
        .token_count = result.generated_tokens,
        .prompt_tokens = result.prompt_tokens,
        .completion_tokens = result.generated_tokens,
        .prefill_ns = result.prefill_ns,
        .generation_ns = result.decode_ns,
        .error_code = 0,
        .finish_reason = finish_reason,
        .tool_calls = tool_calls,
    };
}

/// Generate using HttpEngine (remote OpenAI-compatible inference).
fn generateWithHttpEngine(
    allocator: std.mem.Allocator,
    chat: *Chat,
    http_engine: *HttpEngine,
    config: ?*const CGenerateConfig,
) GenerateResult {
    log.debug("capi_bridge", "Dispatching to HttpEngine", .{}, @src());

    // Build HTTP generation options from C config
    var http_opts = http_engine_mod.GenerateOptions{};

    if (config) |cfg| {
        if (cfg.max_tokens > 0) http_opts.max_tokens = cfg.max_tokens;
        if (cfg.temperature >= 0) http_opts.temperature = cfg.temperature;
        if (cfg.top_p > 0) http_opts.top_p = cfg.top_p;

        // Pass through tool definitions and choice
        if (cfg.tools_json) |t| http_opts.tools_json = std.mem.sliceTo(t, 0);
        if (cfg.tool_choice) |c| http_opts.tool_choice = std.mem.sliceTo(c, 0);

        // Pass through extra body JSON for provider-specific parameters
        if (cfg.extra_body_json) |extra| {
            http_opts.extra_body_json = std.mem.sliceTo(extra, 0);
        }
    }

    // Call HttpEngine (non-streaming mode only - streaming uses iterator API)
    const start_time = std.time.nanoTimestamp();

    const result = http_engine.generate(chat, http_opts) catch |err| {
        log.warn("capi_bridge", "HttpEngine generate failed", .{ .err = @errorName(err) });
        return .{ .error_code = @intFromEnum(error_codes.ErrorCode.generation_failed) };
    };
    defer result.deinit(allocator);

    const end_time = std.time.nanoTimestamp();
    const generation_ns: u64 = @intCast(end_time - start_time);

    const finish_reason = httpFinishReasonToC(result.finish_reason);

    // Build tool call inputs for commit (if any)
    var tc_inputs_buf: [32]commit_mod.ToolCallInput = undefined; // filled element-by-element in loop below
    const tc_inputs: []const commit_mod.ToolCallInput = if (result.tool_calls.len > 0) blk: {
        // Allocate if more than static buffer
        if (result.tool_calls.len > tc_inputs_buf.len) {
            return .{ .error_code = @intFromEnum(error_codes.ErrorCode.out_of_memory) };
        }
        for (result.tool_calls, 0..) |tc, i| {
            tc_inputs_buf[i] = .{
                .id = tc.id,
                .name = tc.name,
                .arguments = tc.arguments,
            };
        }
        break :blk tc_inputs_buf[0..result.tool_calls.len];
    } else &.{};

    // Commit to conversation via shared path (handles firewall, reasoning, chat items)
    commit_mod.commitGenerationResult(allocator, chat, .{
        .text = result.text,
        .tool_calls = tc_inputs,
        .prompt_tokens = result.prompt_tokens,
        .completion_tokens = result.completion_tokens,
        .generation_ns = generation_ns,
        .finish_reason = finishReasonToString(finish_reason),
    }) catch {
        return .{ .error_code = @intFromEnum(error_codes.ErrorCode.out_of_memory) };
    };

    // Copy text for the returned GenerateResult
    const text = allocator.dupe(u8, result.text) catch {
        return .{ .error_code = @intFromEnum(error_codes.ErrorCode.out_of_memory) };
    };

    // Build tool call refs for the caller (if any)
    var tool_refs: ?[]const local.ToolCallRef = null;
    if (result.tool_calls.len > 0) {
        var refs = allocator.alloc(local.ToolCallRef, result.tool_calls.len) catch {
            allocator.free(text);
            return .{ .error_code = @intFromEnum(error_codes.ErrorCode.out_of_memory) };
        };
        var built_count: usize = 0;

        for (result.tool_calls, 0..) |tc, i| {
            const call_id = allocator.dupe(u8, tc.id) catch {
                freeToolCallRefsPartial(allocator, refs, built_count);
                allocator.free(text);
                return .{ .error_code = @intFromEnum(error_codes.ErrorCode.out_of_memory) };
            };
            const name = allocator.dupe(u8, tc.name) catch {
                allocator.free(call_id);
                freeToolCallRefsPartial(allocator, refs, built_count);
                allocator.free(text);
                return .{ .error_code = @intFromEnum(error_codes.ErrorCode.out_of_memory) };
            };
            const arguments = allocator.dupe(u8, tc.arguments) catch {
                allocator.free(call_id);
                allocator.free(name);
                freeToolCallRefsPartial(allocator, refs, built_count);
                allocator.free(text);
                return .{ .error_code = @intFromEnum(error_codes.ErrorCode.out_of_memory) };
            };

            refs[i] = .{
                .item_index = chat.conv.len() - (result.tool_calls.len - i),
                .call_id = call_id,
                .name = name,
                .arguments = arguments,
            };
            built_count += 1;
        }
        tool_refs = refs;
    }

    return .{
        .text = text,
        .token_count = result.completion_tokens,
        .prompt_tokens = result.prompt_tokens,
        .completion_tokens = result.completion_tokens,
        .prefill_ns = 0, // Remote API doesn't report prefill time
        .generation_ns = generation_ns,
        .finish_reason = finish_reason,
        .tool_calls = tool_refs,
        .error_code = 0,
    };
}

// =============================================================================
// Remote Model Listing API
// =============================================================================

/// C-compatible model info for remote endpoints.
pub const CRemoteModelInfo = extern struct {
    /// Model ID (null-terminated, owned)
    id: ?[*:0]u8,
    /// Object type (usually "model")
    object: ?[*:0]u8,
    /// Creation timestamp (0 if not available)
    created: i64,
    /// Owner/organization
    owned_by: ?[*:0]u8,
};

/// C-compatible result from listing remote models.
pub const CRemoteModelListResult = extern struct {
    /// Array of model info (owned)
    models: ?[*]CRemoteModelInfo,
    /// Number of models
    count: usize,
    /// Error code (0 = success)
    error_code: i32,
};

/// List models from a remote OpenAI-compatible backend.
///
/// Caller owns the returned result and must call freeModelListResult() to free it.
pub fn listModels(allocator: std.mem.Allocator, backend: *InferenceBackend) CRemoteModelListResult {
    // Get HTTP engine from backend
    const http_engine = backend.getHttpEngine() orelse {
        var ret = std.mem.zeroes(CRemoteModelListResult);
        ret.error_code = @intFromEnum(error_codes.ErrorCode.invalid_argument);
        return ret;
    };

    // List models via HTTP engine
    var result = http_engine.listModels() catch |err| {
        log.warn("capi_bridge", "listModels failed", .{ .err = @errorName(err) });
        var ret = std.mem.zeroes(CRemoteModelListResult);
        ret.error_code = @intFromEnum(error_codes.errorToCode(err));
        return ret;
    };
    defer result.deinit(allocator);

    // Convert to C format
    return convertModelListResult(allocator, result);
}

/// Convert HttpEngine.ListModelsResult to C-compatible CRemoteModelListResult.
///
/// This is extracted from listModels for testability. The HTTP call is separate
/// from the response conversion logic. All parsing logic tested via mock tests below.
pub fn convertModelListResult(allocator: std.mem.Allocator, result: http_engine_mod.ListModelsResult) CRemoteModelListResult {
    const count = result.models.len;
    if (count == 0) {
        return std.mem.zeroes(CRemoteModelListResult);
    }

    const models = allocator.alloc(CRemoteModelInfo, count) catch {
        var ret = std.mem.zeroes(CRemoteModelListResult);
        ret.error_code = @intFromEnum(error_codes.ErrorCode.out_of_memory);
        return ret;
    };

    for (result.models, 0..) |model, i| {
        var info = std.mem.zeroes(CRemoteModelInfo);
        info.id = dupeToSentinelOrNull(allocator, model.id);
        info.object = dupeToSentinelOrNull(allocator, model.object);
        info.created = model.created orelse 0;
        info.owned_by = dupeToSentinelOrNull(allocator, model.owned_by);
        models[i] = info;
    }

    var ret = std.mem.zeroes(CRemoteModelListResult);
    ret.models = models.ptr;
    ret.count = count;
    return ret;
}

/// Free result from listModels().
pub fn freeModelListResult(allocator: std.mem.Allocator, result: *CRemoteModelListResult) void {
    if (result.models) |models| {
        const slice = models[0..result.count];

        for (slice) |*model| {
            if (model.id) |id| freeSentinelString(allocator, id);
            if (model.object) |obj| freeSentinelString(allocator, obj);
            if (model.owned_by) |owner| freeSentinelString(allocator, owner);
        }

        allocator.free(slice);
        result.models = null;
        result.count = 0;
    }
}

/// Duplicate a slice to a null-terminated C string, or return null if empty.
fn dupeToSentinelOrNull(allocator: std.mem.Allocator, src: []const u8) ?[*:0]u8 {
    if (src.len == 0) return null;
    const buf = allocator.allocSentinel(u8, src.len, 0) catch return null;
    @memcpy(buf, src);
    return buf.ptr;
}

/// Free a null-terminated string allocated with allocSentinel.
fn freeSentinelString(allocator: std.mem.Allocator, ptr: [*:0]u8) void {
    const span = std.mem.span(ptr);
    allocator.free(span.ptr[0 .. span.len + 1]);
}

// =============================================================================
// Public Helpers
// =============================================================================

/// Convert C config to GenerateOptions (basic fields only, no allocations).
///
/// This is a lightweight conversion for the iterator API. It copies scalar
/// fields and string slices (which point to C-owned memory). It does NOT
/// tokenize stop sequences or convert logit bias - those require allocation.
pub fn configToGenerateOptions(config: ?*const CGenerateConfig) GenerateOptions {
    var opts = GenerateOptions{};

    const cfg = config orelse return opts;

    if (cfg.max_tokens > 0) opts.max_tokens = cfg.max_tokens;
    if (cfg.temperature >= 0) opts.temperature = cfg.temperature;
    if (cfg.top_k > 0) opts.top_k = cfg.top_k;
    if (cfg.top_p >= 0) opts.top_p = cfg.top_p;
    if (cfg.min_p >= 0) opts.min_p = cfg.min_p;
    if (cfg.repetition_penalty > 0) opts.repetition_penalty = cfg.repetition_penalty;
    if (cfg.seed != 0) opts.seed = cfg.seed;

    if (cfg.template_override) |t| opts.template_override = std.mem.sliceTo(t, 0);
    if (cfg.extra_context_json) |j| opts.extra_context_json = std.mem.sliceTo(j, 0);
    if (cfg.tools_json) |t| opts.tools_json = std.mem.sliceTo(t, 0);
    if (cfg.tool_choice) |c| opts.tool_choice = std.mem.sliceTo(c, 0);
    opts.raw_output = cfg.raw_output != 0;

    // Pass through stop flag for cancellation support
    opts.stop_flag = cfg.stop_flag;

    // Pass through prefill progress callback
    opts.prefill_progress_fn = cfg.prefill_progress_fn;
    opts.prefill_progress_data = cfg.prefill_progress_data;

    return opts;
}

/// Convert C config to HTTP GenerateOptions (for remote backends).
///
/// Like `configToGenerateOptions` but produces `http_engine_mod.GenerateOptions`.
/// Copies scalar fields and string slice pointers (which reference C-owned memory).
pub fn configToHttpGenerateOptions(config: ?*const CGenerateConfig) http_engine_mod.GenerateOptions {
    var opts = http_engine_mod.GenerateOptions{};

    const cfg = config orelse return opts;

    if (cfg.max_tokens > 0) opts.max_tokens = cfg.max_tokens;
    if (cfg.temperature >= 0) opts.temperature = cfg.temperature;
    if (cfg.top_p > 0) opts.top_p = cfg.top_p;

    if (cfg.tools_json) |t| opts.tools_json = std.mem.sliceTo(t, 0);
    if (cfg.tool_choice) |c| opts.tool_choice = std.mem.sliceTo(c, 0);
    if (cfg.extra_body_json) |extra| opts.extra_body_json = std.mem.sliceTo(extra, 0);
    opts.raw_output = cfg.raw_output != 0;

    return opts;
}

/// Error type for iterator creation.
pub const CreateIteratorError = error{
    InvalidArgument,
    OutOfMemory,
    IteratorCreationFailed,
};

/// Creates a token iterator for pull-based streaming generation.
///
/// This is the core implementation called by the C API. It handles:
/// - Validation of all input parameters
/// - Building content from multipart input
/// - Adding user message to chat
/// - Creating the background generation iterator
pub fn createIterator(
    allocator: std.mem.Allocator,
    chat: *Chat,
    content_parts: []const GenerateContentPart,
    backend_ptr: *InferenceBackend,
    config: ?*const CGenerateConfig,
) CreateIteratorError!*iterator_mod.TokenIterator {
    log.debug("router", "createIterator", .{
        .parts = content_parts.len,
        .conv_items = chat.conv.len(),
        .backend = @as(u8, switch (backend_ptr.backend) {
            .Local => 0,
            .OpenAICompatible => 1,
            .Unspecified => 2,
        }),
    }, @src());

    if (content_parts.len > 0) {
        appendUserMessageFromParts(chat, content_parts) catch {
            return error.OutOfMemory;
        };
    }

    // Dispatch based on backend type
    switch (backend_ptr.backend) {
        .Local => |local_engine| {
            const opts = configToGenerateOptions(config);
            return iterator_mod.TokenIterator.init(allocator, local_engine, chat, opts) catch {
                return error.IteratorCreationFailed;
            };
        },
        .OpenAICompatible => |http_engine| {
            const opts = configToHttpGenerateOptions(config);
            return iterator_mod.TokenIterator.initWithHttpEngine(allocator, http_engine, chat, opts) catch {
                return error.IteratorCreationFailed;
            };
        },
        .Unspecified => return error.InvalidArgument,
    }
}

// =============================================================================
// Internal Implementation
// =============================================================================

/// Built options with allocated resources that need cleanup.
const BuiltOptions = struct {
    options: GenerateOptions,
    stop_sequences: [][]u32,
    logit_bias: []sampler.LogitBiasEntry,

    fn deinit(self: *BuiltOptions, allocator: std.mem.Allocator) void {
        for (self.stop_sequences) |seq| allocator.free(seq);
        if (self.stop_sequences.len > 0) allocator.free(self.stop_sequences);
        if (self.logit_bias.len > 0) allocator.free(self.logit_bias);
    }
};

/// Build GenerateOptions from C config.
fn buildOptions(
    allocator: std.mem.Allocator,
    config: ?*const CGenerateConfig,
    engine: *LocalEngine,
) !BuiltOptions {
    var result = BuiltOptions{
        .options = .{},
        .stop_sequences = &.{},
        .logit_bias = &.{},
    };

    const cfg = config orelse return result;

    if (cfg.max_tokens > 0) result.options.max_tokens = cfg.max_tokens;
    if (cfg.temperature >= 0) result.options.temperature = cfg.temperature;
    if (cfg.top_k > 0) result.options.top_k = cfg.top_k;
    if (cfg.top_p >= 0) result.options.top_p = cfg.top_p;
    if (cfg.min_p >= 0) result.options.min_p = cfg.min_p;
    if (cfg.repetition_penalty > 0) result.options.repetition_penalty = cfg.repetition_penalty;
    if (cfg.seed != 0) result.options.seed = cfg.seed;

    if (cfg.template_override) |t| result.options.template_override = std.mem.sliceTo(t, 0);
    if (cfg.extra_context_json) |j| result.options.extra_context_json = std.mem.sliceTo(j, 0);
    if (cfg.tools_json) |t| result.options.tools_json = std.mem.sliceTo(t, 0);
    if (cfg.tool_choice) |c| result.options.tool_choice = std.mem.sliceTo(c, 0);
    result.options.raw_output = cfg.raw_output != 0;

    // Pass through stop flag for cancellation support
    result.options.stop_flag = cfg.stop_flag;

    // Tokenize stop sequences
    if (cfg.stop_sequences != null and cfg.stop_sequence_count > 0) {
        const seqs = cfg.stop_sequences.?[0..cfg.stop_sequence_count];
        result.stop_sequences = try allocator.alloc([]u32, seqs.len);
        for (seqs, 0..) |seq_ptr, i| {
            result.stop_sequences[i] = try engine.encode(std.mem.sliceTo(seq_ptr, 0));
        }
        result.options.stop_sequences = @as([]const []const u32, result.stop_sequences);
    }

    // Convert logit bias
    if (cfg.logit_bias != null and cfg.logit_bias_count > 0) {
        const entries = cfg.logit_bias.?[0..cfg.logit_bias_count];
        result.logit_bias = try allocator.alloc(sampler.LogitBiasEntry, entries.len);
        for (entries, 0..) |e, i| {
            result.logit_bias[i] = .{ .token_id = e.token_id, .bias = e.bias };
        }
        result.options.logit_bias = result.logit_bias;
    }

    return result;
}

// =============================================================================
// Tests
// =============================================================================

test "contentTypeFromGeneratePart maps known and unknown values" {
    try std.testing.expectEqual(ContentType.input_text, contentTypeFromGeneratePart(0));
    try std.testing.expectEqual(ContentType.input_image, contentTypeFromGeneratePart(1));
    try std.testing.expectEqual(ContentType.input_audio, contentTypeFromGeneratePart(2));
    try std.testing.expectEqual(ContentType.input_video, contentTypeFromGeneratePart(3));
    try std.testing.expectEqual(ContentType.input_file, contentTypeFromGeneratePart(4));
    try std.testing.expectEqual(ContentType.unknown, contentTypeFromGeneratePart(99));
}

test "appendUserMessageFromParts preserves non-text content parts" {
    var chat = try Chat.init(std.testing.allocator);
    defer chat.deinit();

    const image_url = "data:image/jpeg;base64,/9j/4AAQSk";
    const prompt_text = "describe this image";
    const mime: [*:0]const u8 = "image/jpeg";

    const parts = [_]GenerateContentPart{
        .{
            .content_type = 1,
            .data_ptr = image_url.ptr,
            .data_len = image_url.len,
            .mime_ptr = mime,
        },
        .{
            .content_type = 0,
            .data_ptr = prompt_text.ptr,
            .data_len = prompt_text.len,
            .mime_ptr = null,
        },
    };

    try appendUserMessageFromParts(&chat, &parts);

    try std.testing.expectEqual(@as(usize, 1), chat.len());
    const item = chat.get(0).?;
    const msg = item.asMessage().?;
    try std.testing.expectEqual(@as(usize, 2), msg.partCount());

    const part0 = msg.getPart(0).?;
    try std.testing.expectEqual(ContentType.input_image, part0.getContentType());
    try std.testing.expectEqualStrings(image_url, part0.getData());

    const part1 = msg.getPart(1).?;
    try std.testing.expectEqual(ContentType.input_text, part1.getContentType());
    try std.testing.expectEqualStrings(prompt_text, part1.getData());
}

test "CGenerateConfig defaults" {
    // Test Zig default initialization (what Zig code gets)
    const cfg: CGenerateConfig = .{};
    try std.testing.expectEqual(@as(usize, 0), cfg.max_tokens);
    try std.testing.expectEqual(@as(f32, -1.0), cfg.temperature);
    try std.testing.expectEqual(@as(usize, 0), cfg.top_k);
    try std.testing.expectEqual(@as(f32, -1.0), cfg.top_p);
    try std.testing.expectEqual(@as(f32, -1.0), cfg.min_p);
    try std.testing.expectEqual(@as(f32, 0.0), cfg.repetition_penalty);
    try std.testing.expect(cfg.stop_sequences == null);
    try std.testing.expectEqual(@as(usize, 0), cfg.stop_sequence_count);
    try std.testing.expect(cfg.logit_bias == null);
    try std.testing.expectEqual(@as(usize, 0), cfg.logit_bias_count);
    try std.testing.expectEqual(@as(u64, 0), cfg.seed);
    try std.testing.expect(cfg.template_override == null);
    try std.testing.expect(cfg.extra_context_json == null);
    try std.testing.expect(cfg.extra_body_json == null);
    try std.testing.expectEqual(@as(u8, 0), cfg.raw_output);
}

test "GenerateContentPart struct layout" {
    var part = std.mem.zeroes(GenerateContentPart);
    part.content_type = 0;
    part.data_ptr = "hello".ptr;
    part.data_len = 5;
    part.mime_ptr = null;
    try std.testing.expectEqual(@as(u8, 0), part.content_type);
    try std.testing.expectEqual(@as(usize, 5), part.data_len);
    try std.testing.expect(part.mime_ptr == null);
}

test "CLogitBiasEntry struct layout" {
    var entry = std.mem.zeroes(CLogitBiasEntry);
    entry.token_id = 42;
    entry.bias = -100.0;
    try std.testing.expectEqual(@as(u32, 42), entry.token_id);
    try std.testing.expectEqual(@as(f32, -100.0), entry.bias);
}

test "GenerateResult defaults" {
    const result = GenerateResult{};
    try std.testing.expect(result.text == null);
    try std.testing.expectEqual(@as(usize, 0), result.token_count);
    try std.testing.expectEqual(@as(usize, 0), result.prompt_tokens);
    try std.testing.expectEqual(@as(usize, 0), result.completion_tokens);
    try std.testing.expectEqual(@as(u64, 0), result.prefill_ns);
    try std.testing.expectEqual(@as(u64, 0), result.generation_ns);
    try std.testing.expectEqual(@as(i32, 0), result.error_code);
}

test "GenerateResult with error" {
    const result = GenerateResult{ .error_code = @intFromEnum(error_codes.ErrorCode.model_not_found) };
    try std.testing.expect(result.text == null);
    try std.testing.expectEqual(@as(i32, 100), result.error_code);
}

test "GenerateResult deinit frees text" {
    const allocator = std.testing.allocator;
    const text = try allocator.dupe(u8, "test output");

    var result = GenerateResult{
        .text = text,
        .token_count = 5,
        .prompt_tokens = 2,
        .completion_tokens = 5,
        .error_code = 0,
    };

    result.deinit(allocator);
    // No leak = success
}

// -----------------------------------------------------------------------------
// Network-dependent function coverage (signature verification)
//
// The following functions require LocalEngine/HttpEngine and cannot be unit
// tested without a real model or remote server. Their signatures are verified
// here; behavior is covered through:
//
// - generate, generateWithBackend: integration tests in bindings/python/tests/
// - listModels: integration tests in bindings/python/tests/model/test_remote_backend.py
// -----------------------------------------------------------------------------

test "generate: signature verification" {
    // Verify function signature compiles correctly.
    // Cannot unit test: requires LocalEngine with loaded model.
    const F = @TypeOf(generate);
    const info = @typeInfo(F).@"fn";
    try std.testing.expectEqual(@as(usize, 6), info.params.len);
}

test "generateWithBackend: signature verification" {
    // Verify function signature compiles correctly.
    // Cannot unit test: requires InferenceBackend with loaded model.
    const F = @TypeOf(generateWithBackend);
    const info = @typeInfo(F).@"fn";
    try std.testing.expectEqual(@as(usize, 5), info.params.len);
}

test "listModels: signature verification" {
    // Verify function signature compiles correctly.
    // Cannot unit test: requires HttpEngine connected to remote server.
    const F = @TypeOf(listModels);
    const info = @typeInfo(F).@"fn";
    try std.testing.expectEqual(@as(usize, 2), info.params.len);
}

// -----------------------------------------------------------------------------
// Conversion and utility function tests (can be unit tested)
// -----------------------------------------------------------------------------

test "toCResult: converts success result with text" {
    const allocator = std.testing.allocator;

    const text = try allocator.dupe(u8, "hello world");
    const input = GenerateResult{
        .text = text,
        .token_count = 5,
        .prompt_tokens = 2,
        .completion_tokens = 5,
        .prefill_ns = 1000,
        .generation_ns = 2000,
        .error_code = 0,
    };

    const c_result = toCResult(allocator, input);
    defer {
        // Free the C string
        if (c_result.text) |t| {
            allocator.free(t[0 .. std.mem.len(t) + 1]);
        }
    }

    try std.testing.expect(c_result.text != null);
    try std.testing.expectEqualStrings("hello world", std.mem.span(c_result.text.?));
    try std.testing.expectEqual(@as(usize, 5), c_result.token_count);
    try std.testing.expectEqual(@as(usize, 2), c_result.prompt_tokens);
    try std.testing.expectEqual(@as(usize, 5), c_result.completion_tokens);
    try std.testing.expectEqual(@as(u64, 1000), c_result.prefill_ns);
    try std.testing.expectEqual(@as(u64, 2000), c_result.generation_ns);
    try std.testing.expectEqual(@as(i32, 0), c_result.error_code);
}

test "toCResult: handles error result" {
    const allocator = std.testing.allocator;

    const input = GenerateResult{
        .text = null,
        .error_code = 100, // model_not_found
    };

    const c_result = toCResult(allocator, input);

    try std.testing.expect(c_result.text == null);
    try std.testing.expectEqual(@as(i32, 100), c_result.error_code);
}

test "toCResult: handles null text as error" {
    const allocator = std.testing.allocator;

    const input = GenerateResult{
        .text = null,
        .error_code = 0, // No error code but no text = internal error
    };

    const c_result = toCResult(allocator, input);

    try std.testing.expect(c_result.text == null);
    try std.testing.expectEqual(@intFromEnum(error_codes.ErrorCode.internal_error), c_result.error_code);
}

test "freeResult: frees text correctly" {
    const allocator = std.testing.allocator;

    const cstr = try allocator.allocSentinel(u8, 5, 0);
    @memcpy(cstr, "hello");

    var result = std.mem.zeroes(CGenerateResult);
    result.text = cstr.ptr;
    result.token_count = 1;
    result.completion_tokens = 1;

    freeResult(allocator, &result);

    try std.testing.expect(result.text == null);
}

test "freeResult: handles null text" {
    const allocator = std.testing.allocator;

    var result = std.mem.zeroes(CGenerateResult);
    result.error_code = 100;

    freeResult(allocator, &result);
    // Should not crash
    try std.testing.expect(result.text == null);
}

test "dupeToSentinelOrNull: duplicates non-empty string" {
    const allocator = std.testing.allocator;

    const result = dupeToSentinelOrNull(allocator, "hello");
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("hello", std.mem.span(result.?));

    freeSentinelString(allocator, result.?);
}

test "dupeToSentinelOrNull: returns null for empty string" {
    const allocator = std.testing.allocator;

    const result = dupeToSentinelOrNull(allocator, "");
    try std.testing.expect(result == null);
}

test "freeSentinelString: frees correctly" {
    const allocator = std.testing.allocator;

    const str = try allocator.allocSentinel(u8, 4, 0);
    @memcpy(str, "test");

    freeSentinelString(allocator, str.ptr);
    // No leak = success
}

test "CRemoteModelInfo struct layout" {
    var info = std.mem.zeroes(CRemoteModelInfo);
    info.created = 1234567890;
    try std.testing.expect(info.id == null);
    try std.testing.expectEqual(@as(i64, 1234567890), info.created);
}

test "CRemoteModelListResult struct layout" {
    const result = std.mem.zeroes(CRemoteModelListResult);
    try std.testing.expect(result.models == null);
    try std.testing.expectEqual(@as(usize, 0), result.count);
    try std.testing.expectEqual(@as(i32, 0), result.error_code);
}

test "freeModelListResult: handles empty result" {
    const allocator = std.testing.allocator;

    var result = std.mem.zeroes(CRemoteModelListResult);

    freeModelListResult(allocator, &result);
    try std.testing.expect(result.models == null);
}

test "freeModelListResult: frees models array" {
    const allocator = std.testing.allocator;

    // Allocate a models array
    var models = try allocator.alloc(CRemoteModelInfo, 2);
    errdefer allocator.free(models);

    const id0 = try allocator.allocSentinel(u8, 5, 0);
    errdefer allocator.free(id0);
    @memcpy(id0, "gpt-4");

    const obj0 = try allocator.allocSentinel(u8, 5, 0);
    errdefer allocator.free(obj0);
    @memcpy(obj0, "model");

    const owner0 = try allocator.allocSentinel(u8, 6, 0);
    errdefer allocator.free(owner0);
    @memcpy(owner0, "openai");

    models[0] = std.mem.zeroes(CRemoteModelInfo);
    models[0].id = id0.ptr;
    models[0].object = obj0.ptr;
    models[0].created = 12345;
    models[0].owned_by = owner0.ptr;

    const id1 = try allocator.allocSentinel(u8, 8, 0);
    @memcpy(id1, "gpt-4o-m");

    models[1] = std.mem.zeroes(CRemoteModelInfo);
    models[1].id = id1.ptr;
    // Test null handling: object and owned_by remain null from zeroes

    var result = std.mem.zeroes(CRemoteModelListResult);
    result.models = models.ptr;
    result.count = 2;

    freeModelListResult(allocator, &result);
    try std.testing.expect(result.models == null);
    try std.testing.expectEqual(@as(usize, 0), result.count);
}

// -----------------------------------------------------------------------------
// Response conversion tests (mock API responses)
//
// These tests verify conversion logic by constructing mock HttpEngine results
// and passing them to the conversion function. No HTTP calls are made.
// -----------------------------------------------------------------------------

test "convertModelListResult: mock response with multiple models" {
    const allocator = std.testing.allocator;

    // Create mock ModelInfo array (simulating what HttpEngine.parseModelsResponse returns)
    var mock_models = try allocator.alloc(http_engine_mod.ModelInfo, 3);
    defer allocator.free(mock_models);

    // Note: we don't allocate these strings - they're "borrowed" from the mock response
    // In real usage, HttpEngine.parseModelsResponse allocates them
    mock_models[0] = .{
        .id = "gpt-4o",
        .object = "model",
        .created = 1686935002,
        .owned_by = "openai",
    };
    mock_models[1] = .{
        .id = "gpt-4o-mini",
        .object = "model",
        .created = 1721172741,
        .owned_by = "system",
    };
    mock_models[2] = .{
        .id = "org/model-name",
        .object = "model",
        .created = null, // Some providers don't have created timestamp
        .owned_by = "vllm",
    };

    const mock_result = http_engine_mod.ListModelsResult{
        .models = mock_models,
    };

    // Convert to C format
    var c_result = convertModelListResult(allocator, mock_result);
    defer freeModelListResult(allocator, &c_result);

    // Verify conversion
    try std.testing.expectEqual(@as(i32, 0), c_result.error_code);
    try std.testing.expectEqual(@as(usize, 3), c_result.count);
    try std.testing.expect(c_result.models != null);

    const models = c_result.models.?[0..c_result.count];

    // Model 0
    try std.testing.expectEqualStrings("gpt-4o", std.mem.span(models[0].id.?));
    try std.testing.expectEqualStrings("model", std.mem.span(models[0].object.?));
    try std.testing.expectEqual(@as(i64, 1686935002), models[0].created);
    try std.testing.expectEqualStrings("openai", std.mem.span(models[0].owned_by.?));

    // Model 1
    try std.testing.expectEqualStrings("gpt-4o-mini", std.mem.span(models[1].id.?));
    try std.testing.expectEqual(@as(i64, 1721172741), models[1].created);

    // Model 2 - null created should become 0
    try std.testing.expectEqualStrings("org/model-name", std.mem.span(models[2].id.?));
    try std.testing.expectEqual(@as(i64, 0), models[2].created);
}

test "convertModelListResult: mock empty response" {
    const allocator = std.testing.allocator;

    const mock_result = http_engine_mod.ListModelsResult{
        .models = &.{},
    };

    var c_result = convertModelListResult(allocator, mock_result);
    defer freeModelListResult(allocator, &c_result);

    try std.testing.expectEqual(@as(i32, 0), c_result.error_code);
    try std.testing.expectEqual(@as(usize, 0), c_result.count);
    try std.testing.expect(c_result.models == null);
}

test "convertModelListResult: mock single model (typical vLLM response)" {
    const allocator = std.testing.allocator;

    var mock_models = try allocator.alloc(http_engine_mod.ModelInfo, 1);
    defer allocator.free(mock_models);

    mock_models[0] = .{
        .id = "my-org/model-7b-chat-hf",
        .object = "model",
        .created = 1234567890,
        .owned_by = "vllm",
    };

    const mock_result = http_engine_mod.ListModelsResult{
        .models = mock_models,
    };

    var c_result = convertModelListResult(allocator, mock_result);
    defer freeModelListResult(allocator, &c_result);

    try std.testing.expectEqual(@as(usize, 1), c_result.count);
    try std.testing.expectEqualStrings("my-org/model-7b-chat-hf", std.mem.span(c_result.models.?[0].id.?));
}

test "convertModelListResult: mock response with empty strings" {
    const allocator = std.testing.allocator;

    var mock_models = try allocator.alloc(http_engine_mod.ModelInfo, 1);
    defer allocator.free(mock_models);

    // Test that empty strings become null (dupeToSentinelOrNull behavior)
    mock_models[0] = .{
        .id = "model-1",
        .object = "",
        .created = null,
        .owned_by = "",
    };

    const mock_result = http_engine_mod.ListModelsResult{
        .models = mock_models,
    };

    var c_result = convertModelListResult(allocator, mock_result);
    defer freeModelListResult(allocator, &c_result);

    try std.testing.expectEqual(@as(usize, 1), c_result.count);
    const model = c_result.models.?[0];
    try std.testing.expectEqualStrings("model-1", std.mem.span(model.id.?));
    try std.testing.expect(model.object == null); // Empty string -> null
    try std.testing.expect(model.owned_by == null); // Empty string -> null
    try std.testing.expectEqual(@as(i64, 0), model.created); // null -> 0
}

// -----------------------------------------------------------------------------
// Tool Call Tests
// -----------------------------------------------------------------------------

test "CToolCallRef struct layout" {
    var ref = std.mem.zeroes(CToolCallRef);
    ref.item_index = 42;
    try std.testing.expectEqual(@as(usize, 42), ref.item_index);
    try std.testing.expect(ref.call_id == null);
    try std.testing.expect(ref.name == null);
    try std.testing.expect(ref.arguments == null);
}

test "CFinishReason enum values" {
    try std.testing.expectEqual(@as(u8, 0), @intFromEnum(CFinishReason.eos_token));
    try std.testing.expectEqual(@as(u8, 1), @intFromEnum(CFinishReason.length));
    try std.testing.expectEqual(@as(u8, 2), @intFromEnum(CFinishReason.stop_sequence));
    try std.testing.expectEqual(@as(u8, 3), @intFromEnum(CFinishReason.tool_calls));
    try std.testing.expectEqual(@as(u8, 4), @intFromEnum(CFinishReason.content_filter));
}

test "toCResult: converts result with tool calls" {
    const allocator = std.testing.allocator;

    // Create mock tool calls
    var tool_calls = try allocator.alloc(local.ToolCallRef, 2);
    defer allocator.free(tool_calls);

    const call_id_1 = try allocator.dupe(u8, "call_abc123");
    defer allocator.free(call_id_1);
    const name_1 = try allocator.dupe(u8, "get_weather");
    defer allocator.free(name_1);
    const args_1 = try allocator.dupe(u8, "{\"location\":\"Paris\"}");
    defer allocator.free(args_1);

    const call_id_2 = try allocator.dupe(u8, "call_def456");
    defer allocator.free(call_id_2);
    const name_2 = try allocator.dupe(u8, "search");
    defer allocator.free(name_2);
    const args_2 = try allocator.dupe(u8, "{\"query\":\"zig\"}");
    defer allocator.free(args_2);

    tool_calls[0] = local.ToolCallRef{
        .item_index = 5,
        .call_id = call_id_1,
        .name = name_1,
        .arguments = args_1,
    };
    tool_calls[1] = local.ToolCallRef{
        .item_index = 6,
        .call_id = call_id_2,
        .name = name_2,
        .arguments = args_2,
    };

    const text = try allocator.dupe(u8, "tool call output");
    const input = GenerateResult{
        .text = text,
        .token_count = 10,
        .prompt_tokens = 4,
        .completion_tokens = 10,
        .prefill_ns = 1000,
        .generation_ns = 2000,
        .error_code = 0,
        .finish_reason = .tool_calls,
        .tool_calls = tool_calls,
    };

    var c_result = toCResult(allocator, input);
    defer freeResult(allocator, &c_result);

    // Verify basic fields
    try std.testing.expect(c_result.text != null);
    try std.testing.expectEqualStrings("tool call output", std.mem.span(c_result.text.?));
    try std.testing.expectEqual(@as(usize, 10), c_result.token_count);
    try std.testing.expectEqual(@as(i32, 0), c_result.error_code);

    // Verify finish reason
    try std.testing.expectEqual(@as(u8, 3), c_result.finish_reason); // tool_calls

    // Verify tool calls
    try std.testing.expectEqual(@as(usize, 2), c_result.tool_call_count);
    try std.testing.expect(c_result.tool_calls != null);

    const calls = c_result.tool_calls.?[0..c_result.tool_call_count];
    try std.testing.expectEqual(@as(usize, 5), calls[0].item_index);
    try std.testing.expectEqualStrings("call_abc123", std.mem.span(calls[0].call_id.?));
    try std.testing.expectEqualStrings("get_weather", std.mem.span(calls[0].name.?));
    try std.testing.expectEqualStrings("{\"location\":\"Paris\"}", std.mem.span(calls[0].arguments.?));

    try std.testing.expectEqual(@as(usize, 6), calls[1].item_index);
    try std.testing.expectEqualStrings("call_def456", std.mem.span(calls[1].call_id.?));
    try std.testing.expectEqualStrings("search", std.mem.span(calls[1].name.?));
    try std.testing.expectEqualStrings("{\"query\":\"zig\"}", std.mem.span(calls[1].arguments.?));
}

test "toCResult: converts result without tool calls" {
    const allocator = std.testing.allocator;

    const text = try allocator.dupe(u8, "regular output");
    const input = GenerateResult{
        .text = text,
        .token_count = 5,
        .prompt_tokens = 2,
        .completion_tokens = 5,
        .prefill_ns = 500,
        .generation_ns = 1000,
        .error_code = 0,
        .finish_reason = .eos_token,
        .tool_calls = null,
    };

    var c_result = toCResult(allocator, input);
    defer freeResult(allocator, &c_result);

    try std.testing.expect(c_result.text != null);
    try std.testing.expectEqual(@as(u8, 0), c_result.finish_reason); // eos_token
    try std.testing.expectEqual(@as(usize, 0), c_result.tool_call_count);
    try std.testing.expect(c_result.tool_calls == null);
}

test "freeResult: frees tool calls correctly" {
    const allocator = std.testing.allocator;

    // Manually create C tool calls (simulating what toCResult does)
    const c_calls = try allocator.alloc(CToolCallRef, 1);
    errdefer allocator.free(c_calls);

    const cstr_text = try allocator.allocSentinel(u8, 4, 0);
    errdefer allocator.free(cstr_text);
    @memcpy(cstr_text, "test");

    const call_id = try allocator.allocSentinel(u8, 8, 0);
    errdefer allocator.free(call_id);
    @memcpy(call_id, "call_123");

    const name = try allocator.allocSentinel(u8, 11, 0);
    @memcpy(name, "get_weather");
    const args = try allocator.allocSentinel(u8, 17, 0);
    @memcpy(args, "{\"location\":\"NY\"}");

    c_calls[0] = std.mem.zeroes(CToolCallRef);
    c_calls[0].call_id = call_id.ptr;
    c_calls[0].name = name.ptr;
    c_calls[0].arguments = args.ptr;

    var result = std.mem.zeroes(CGenerateResult);
    result.text = cstr_text.ptr;
    result.token_count = 1;
    result.completion_tokens = 1;
    result.finish_reason = @intFromEnum(CFinishReason.tool_calls);
    result.tool_calls = c_calls.ptr;
    result.tool_call_count = 1;

    freeResult(allocator, &result);

    // All pointers should be null after free
    try std.testing.expect(result.text == null);
    try std.testing.expect(result.tool_calls == null);
    try std.testing.expectEqual(@as(usize, 0), result.tool_call_count);
}

test "CGenerateConfig: tools_json and tool_choice fields" {
    const cfg = std.mem.zeroes(CGenerateConfig);
    try std.testing.expect(cfg.tools_json == null);
    try std.testing.expect(cfg.tool_choice == null);

    var cfg2 = std.mem.zeroes(CGenerateConfig);
    cfg2.tools_json = "tools";
    cfg2.tool_choice = "auto";
    try std.testing.expect(cfg2.tools_json != null);
    try std.testing.expectEqualStrings("tools", std.mem.sliceTo(cfg2.tools_json.?, 0));
    try std.testing.expectEqualStrings("auto", std.mem.sliceTo(cfg2.tool_choice.?, 0));
    cfg2.raw_output = 1;
    try std.testing.expectEqual(@as(u8, 1), cfg2.raw_output);
}
