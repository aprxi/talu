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
const responses_mod = @import("../responses/root.zig");
const inference_bridge = @import("inference_bridge.zig");
const inference_types = inference_bridge.types;
const vision_types = inference_bridge.vision_types;
const sampler = inference_bridge.sampling;
const tokenizer_mod = @import("../tokenizer/root.zig");
const error_codes = @import("../capi/error_codes.zig");
const capi_error = @import("../capi/error.zig");
const log = @import("log_pkg");

const LocalEngine = local.LocalEngine;
const InferenceBackend = spec.InferenceBackend;
const GenerateOptions = local.GenerateOptions;
const Chat = responses_mod.Chat;
const ContentType = responses_mod.ContentType;
const FinishReason = inference_types.FinishReason;

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

/// Preprocessed vision image payload from C API callers.
///
/// `pixels_ptr` points to a contiguous array of `pixel_count` f32 values
/// (row-major CHW layout expected by inference backends).
pub const CGenerateVisionImage = extern struct {
    pixels_ptr: ?[*]const f32 = null,
    pixel_count: usize = 0,
    width: u32 = 0,
    height: u32 = 0,
    grid_temporal: u32 = 0,
    grid_height: u32 = 0,
    grid_width: u32 = 0,
    token_count: usize = 0,
};

/// Generation configuration from C API.
pub const CGenerateConfig = extern struct {
    max_tokens: usize = 0,
    /// Maximum tokens for the answer/completion only (0 = unlimited).
    max_completion_tokens: usize = 0,
    /// Maximum thinking/reasoning tokens. maxInt = unset (use effort), 0 = no thinking.
    max_reasoning_tokens: usize = std.math.maxInt(usize),
    temperature: f32 = -1.0,
    top_k: usize = 0,
    top_p: f32 = -1.0,
    min_p: f32 = -1.0,
    repetition_penalty: f32 = -1.0,
    presence_penalty: f32 = -1.0,
    frequency_penalty: f32 = -1.0,
    stop_sequences: ?[*]const [*:0]const u8 = null,
    stop_sequence_count: usize = 0,
    logit_bias: ?[*]const CLogitBiasEntry = null,
    logit_bias_count: usize = 0,
    seed: u64 = 0,
    template_override: ?[*:0]const u8 = null,
    extra_context_json: ?[*:0]const u8 = null,
    /// Reasoning effort level: "none", "low", "medium", "high", "xhigh".
    /// Mapped to template variables (e.g. enable_thinking) by the router.
    reasoning_effort: ?[*:0]const u8 = null,
    /// Tool definitions as JSON array. Format: [{"type":"function","function":{...}}]
    tools_json: ?[*:0]const u8 = null,
    /// Tool choice strategy: "auto", "required", "none", or specific function name.
    tool_choice: ?[*:0]const u8 = null,
    /// Optional stop flag for cancellation. When set to true (non-zero), generation stops.
    /// This allows external cancellation (e.g., client disconnect, asyncio.CancelledError)
    /// without waiting for the next callback invocation.
    /// The pointer must remain valid for the duration of generation.
    stop_flag: ?*const std.atomic.Value(bool) = null,
    /// Reserved for compatibility; ignored by local inference.
    extra_body_json: ?[*:0]const u8 = null,

    /// When non-zero, preserve raw model output bytes without reasoning-tag filtering.
    /// Default (0) keeps current behavior (reasoning tags parsed into typed items).
    raw_output: usize = 0,

    /// When non-zero, behave like a standard completions endpoint: no thinking
    /// intervention, no reasoning separation, just raw token generation with
    /// max_tokens as the sole cap.
    completions_mode: usize = 0,

    /// Optional prefill progress callback. Called once per transformer layer
    /// during prefill (not decode). Signature: fn(completed_layers, total_layers, userdata).
    prefill_progress_fn: ?*const fn (usize, usize, ?*anyopaque) callconv(.c) void = null,
    prefill_progress_data: ?*anyopaque = null,

    /// Optional externally preprocessed vision payload.
    /// When present, local image decode/preprocess is skipped.
    external_vision_images: ?[*]const CGenerateVisionImage = null,
    external_vision_image_count: usize = 0,
    external_vision_image_token_id: u32 = 0,
    _external_vision_padding: u32 = 0,
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

/// Generation result (internal, Zig slice).
pub const GenerateResult = struct {
    text: ?[]const u8 = null,
    token_count: usize = 0,
    prompt_tokens: usize = 0,
    completion_tokens: usize = 0,
    prefill_ns: u64 = 0,
    generation_ns: u64 = 0,
    ttft_ns: u64 = 0,
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
    ttft_ns: u64,
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
        ret.ttft_ns = result.ttft_ns;
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
                ret.ttft_ns = result.ttft_ns;
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
    ret.ttft_ns = result.ttft_ns;
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
    const resolved_id = root.resolveForRouting(model_id) catch |err| return .{
        .error_code = @intFromEnum(switch (err) {
            error.UnsupportedNamespace => error_codes.ErrorCode.ambiguous_backend,
        }),
    };

    // Get or create engine
    const engine = root.getOrCreateEngineWithConfig(allocator, resolved_id, resolution) catch return .{ .error_code = @intFromEnum(error_codes.ErrorCode.model_not_found) };

    // Create user message from content parts
    appendUserMessageFromParts(chat, parts) catch {
        return .{ .error_code = @intFromEnum(error_codes.ErrorCode.out_of_memory) };
    };

    // Local generation calls engine.generate() directly.
    return generateWithLocalEngine(allocator, chat, engine, config);
}

/// Generate using a spec-based InferenceBackend.
///
/// Like generate() but takes an InferenceBackend created from a CanonicalSpec
/// instead of a model ID string. This is the new spec-based API that will replace
/// the string-based API.
///
/// Dispatches to the local backend.
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
            .Unspecified => 1,
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
        .Unspecified => {
            return .{ .error_code = @intFromEnum(error_codes.ErrorCode.invalid_argument) };
        },
    }
}

/// Generate using LocalEngine (native inference, non-streaming).
fn generateWithLocalEngine(
    allocator_: std.mem.Allocator,
    chat: *Chat,
    local_engine: *LocalEngine,
    config: ?*const CGenerateConfig,
) GenerateResult {
    var built = buildOptions(allocator_, config, local_engine) catch |err| {
        capi_error.setError(err, "Invalid generation options", .{});
        return .{ .error_code = @intFromEnum(buildOptionsErrorCode(err)) };
    };
    defer built.deinit(allocator_);

    log.debug("router", "LocalEngine generate (direct route)", .{
        .max_tokens = built.options.max_tokens orelse 0,
        .has_tools = @as(u8, @intFromBool(built.options.tools_json != null)),
    }, @src());

    var gen_result = local_engine.generate(chat, built.options) catch |err| {
        capi_error.setError(err, "Generation failed", .{});
        return .{ .error_code = @intFromEnum(error_codes.ErrorCode.generation_failed) };
    };
    defer gen_result.deinit(local_engine.allocator);

    const finish_reason: CFinishReason = switch (gen_result.finish_reason) {
        .eos_token => .eos_token,
        .length => .length,
        .stop_sequence => .stop_sequence,
        .tool_calls => .tool_calls,
        .content_filter => .content_filter,
        .cancelled => .cancelled,
    };

    // Transfer tool call ownership from gen_result to our return value.
    var tool_calls: ?[]const local.ToolCallRef = null;
    if (gen_result.tool_calls) |tc| {
        tool_calls = tc;
        gen_result.tool_calls = null;
    }

    const result_text = allocator_.dupe(u8, gen_result.text) catch return .{ .error_code = @intFromEnum(error_codes.ErrorCode.out_of_memory) };

    return .{
        .text = result_text,
        .token_count = gen_result.generated_tokens,
        .prompt_tokens = gen_result.prompt_tokens,
        .completion_tokens = gen_result.generated_tokens,
        .prefill_ns = gen_result.prefill_ns,
        .generation_ns = gen_result.decode_ns,
        .ttft_ns = 0,
        .error_code = 0,
        .finish_reason = finish_reason,
        .tool_calls = tool_calls,
    };
}

// =============================================================================
// Streaming Generation (callback-based)
// =============================================================================

/// C callback type for streaming generation.
/// Returns 1 to continue, 0 to stop.
pub const StreamCallback = *const fn ([*]const u8, usize, u8, u8, u8, ?*anyopaque) callconv(.c) u8;

/// Maximum decoded bytes per token.
const MAX_TOKEN_LEN = 512;

/// Per-token streaming wrapper. Held on the stack during generation.
/// Decodes tokens, applies UTF-8 assembly, classifies based on scheduler's
/// `in_thinking` state, then calls the user's rich C callback.
const StreamingWrapper = struct {
    tok: *tokenizer_mod.Tokenizer,
    user_cb: StreamCallback,
    user_data: ?*anyopaque,
    stop_flag: *std.atomic.Value(bool),
    raw_output: bool = false,
    is_tool_generation: bool = false,
    decode_context_token: ?u32 = null,

    // UTF-8 pending bytes
    utf8_pending: [3]u8 = .{ 0, 0, 0 },
    utf8_pending_len: u8 = 0,

    const DecodedToken = struct {
        bytes: []const u8,
        owned: ?[]u8 = null,
    };

    fn longestCommonPrefixLen(a: []const u8, b: []const u8) usize {
        const n = @min(a.len, b.len);
        var i: usize = 0;
        while (i < n and a[i] == b[i]) : (i += 1) {}
        return i;
    }

    /// Decode token bytes for streaming.
    ///
    /// Fast path: prebuilt token byte table.
    /// Fallback: context-aware raw decode delta for tokenizers that require
    /// neighboring context to produce non-empty token text.
    fn decodeTokenBytes(self: *StreamingWrapper, token_id: u32) DecodedToken {
        if (self.tok.tokenBytes(token_id)) |bytes| {
            if (bytes.len > 0) {
                return .{ .bytes = bytes };
            }
        }

        const token_options = tokenizer_mod.Tokenizer.DecodeOptions{
            .skip_special_tokens = true,
        };

        if (self.decode_context_token) |ctx_token| {
            const ctx_raw = self.tok.decodeRawBytes(&[_]u32{ctx_token}, token_options) catch
                return .{ .bytes = "" };
            defer self.tok.allocator.free(ctx_raw);

            const pair_raw = self.tok.decodeRawBytes(
                &[_]u32{ ctx_token, token_id },
                token_options,
            ) catch return .{ .bytes = "" };
            errdefer self.tok.allocator.free(pair_raw);

            const prefix = longestCommonPrefixLen(ctx_raw, pair_raw);
            if (prefix >= ctx_raw.len and prefix < pair_raw.len) {
                const delta = pair_raw[prefix..];
                const out = self.tok.allocator.alloc(u8, delta.len) catch {
                    self.tok.allocator.free(pair_raw);
                    return .{ .bytes = "" };
                };
                @memcpy(out, delta);
                self.tok.allocator.free(pair_raw);
                return .{ .bytes = out, .owned = out };
            }

            self.tok.allocator.free(pair_raw);
        }

        const raw = self.tok.decodeRawBytes(&[_]u32{token_id}, token_options) catch
            return .{ .bytes = "" };
        return .{ .bytes = raw, .owned = raw };
    }

    /// TokenCallback-compatible entry point: fn(token_id, in_thinking, userdata) void.
    fn onToken(token_id: u32, in_thinking: bool, user_data: ?*anyopaque) void {
        const self: *StreamingWrapper = @ptrCast(@alignCast(user_data));

        const decoded_token = self.decodeTokenBytes(token_id);
        defer if (decoded_token.owned) |owned| self.tok.allocator.free(owned);
        const decoded_raw: []const u8 = decoded_token.bytes;
        self.decode_context_token = token_id;
        if (decoded_raw.len == 0) return;

        // UTF-8 assembly (same algorithm as batch.zig).
        var combined_buf: [3 + MAX_TOKEN_LEN]u8 = undefined;
        const pending_len: usize = self.utf8_pending_len;
        @memcpy(combined_buf[0..pending_len], self.utf8_pending[0..pending_len]);
        const raw_copy_len = @min(decoded_raw.len, combined_buf.len - pending_len);
        @memcpy(combined_buf[pending_len..][0..raw_copy_len], decoded_raw[0..raw_copy_len]);
        const total_len = pending_len + raw_copy_len;
        const combined = combined_buf[0..total_len];

        const valid_end = utf8ValidPrefix(combined);
        const valid = combined[0..valid_end];
        const trailing = combined[valid_end..total_len];

        // Store trailing incomplete UTF-8 bytes.
        self.utf8_pending_len = 0;
        if (trailing.len > 0 and trailing.len <= 3) {
            const lead = trailing[0];
            const expected_len: usize = if (lead & 0xE0 == 0xC0)
                2
            else if (lead & 0xF0 == 0xE0)
                3
            else if (lead & 0xF8 == 0xF0)
                4
            else
                0;
            if (expected_len > 0 and trailing.len < expected_len) {
                var all_valid = true;
                for (trailing[1..]) |byte| {
                    if (byte & 0xC0 != 0x80) {
                        all_valid = false;
                        break;
                    }
                }
                if (all_valid) {
                    self.utf8_pending_len = @intCast(trailing.len);
                    @memcpy(self.utf8_pending[0..trailing.len], trailing);
                }
            }
        }

        if (valid.len == 0) return;

        // Classify based on in_thinking parameter from scheduler.
        var item_type: u8 = @intFromEnum(responses_mod.ItemType.message);
        var content_type: u8 = @intFromEnum(ContentType.output_text);
        if (!self.raw_output) {
            if (in_thinking) {
                item_type = @intFromEnum(responses_mod.ItemType.reasoning);
                content_type = @intFromEnum(ContentType.reasoning_text);
            } else if (self.is_tool_generation) {
                item_type = @intFromEnum(responses_mod.ItemType.function_call);
            }
        }

        const cont = self.user_cb(valid.ptr, valid.len, item_type, content_type, 0, self.user_data);
        if (cont == 0) self.stop_flag.store(true, .release);
    }
};

/// Find the boundary of complete UTF-8 codepoints in a byte slice.
fn utf8ValidPrefix(bytes: []const u8) usize {
    var last_valid: usize = 0;
    var i: usize = 0;
    while (i < bytes.len) {
        const b = bytes[i];
        const seq_len: usize = if (b < 0x80)
            1
        else if (b & 0xE0 == 0xC0)
            2
        else if (b & 0xF0 == 0xE0)
            3
        else if (b & 0xF8 == 0xF0)
            4
        else {
            i += 1;
            continue;
        };
        if (i + seq_len > bytes.len) break;
        var valid = true;
        var j: usize = 1;
        while (j < seq_len) : (j += 1) {
            if (bytes[i + j] & 0xC0 != 0x80) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            i += 1;
            continue;
        }
        i += seq_len;
        last_valid = i;
    }
    return last_valid;
}

/// Generate with streaming callback using LocalEngine.
/// Calls engine.generate with a TokenCallback wrapper that decodes tokens,
/// applies reasoning-tag filtering, and invokes the user's rich callback.
fn generateStreamingWithLocalEngine(
    allocator_: std.mem.Allocator,
    chat: *Chat,
    local_engine: *LocalEngine,
    config: ?*const CGenerateConfig,
    stream_cb: StreamCallback,
    stream_cb_data: ?*anyopaque,
) GenerateResult {
    var built = buildOptions(allocator_, config, local_engine) catch |err| {
        capi_error.setError(err, "Invalid generation options", .{});
        return .{ .error_code = @intFromEnum(buildOptionsErrorCode(err)) };
    };
    defer built.deinit(allocator_);

    // Determine tool generation mode.
    const is_tool_generation = built.options.tools_json != null;

    // Raw output mode.
    const raw_output = if (config) |cfg| cfg.raw_output != 0 else false;

    // Create a local stop flag if none provided (for callback→stop signaling).
    var local_stop_flag = std.atomic.Value(bool).init(false);
    const stop_flag_ptr: *std.atomic.Value(bool) = if (built.options.stop_flag) |sf|
        @constCast(sf)
    else
        &local_stop_flag;
    built.options.stop_flag = stop_flag_ptr;

    // Set up streaming wrapper.
    var wrapper = StreamingWrapper{
        .tok = &local_engine.tok,
        .user_cb = stream_cb,
        .user_data = stream_cb_data,
        .stop_flag = stop_flag_ptr,
        .raw_output = raw_output,
        .is_tool_generation = is_tool_generation,
    };

    // Wire the token callback.
    built.options.token_callback = StreamingWrapper.onToken;
    built.options.callback_data = @ptrCast(&wrapper);

    log.debug("router", "LocalEngine generate (streaming callback route)", .{
        .max_tokens = built.options.max_tokens orelse 0,
        .has_tools = @as(u8, @intFromBool(is_tool_generation)),
    }, @src());

    // Run synchronous generation. Token callback fires per token.
    var gen_result = local_engine.generate(chat, built.options) catch |err| {
        capi_error.setError(err, "Generation failed", .{});
        return .{ .error_code = @intFromEnum(error_codes.ErrorCode.generation_failed) };
    };
    defer gen_result.deinit(local_engine.allocator);

    // Send final callback (empty text, is_final=1).
    _ = stream_cb("".ptr, 0, @intFromEnum(responses_mod.ItemType.message), @intFromEnum(ContentType.output_text), 1, stream_cb_data);

    // Map finish reason.
    const finish_reason: CFinishReason = switch (gen_result.finish_reason) {
        .eos_token => .eos_token,
        .length => .length,
        .stop_sequence => .stop_sequence,
        .tool_calls => .tool_calls,
        .content_filter => .content_filter,
        .cancelled => .cancelled,
    };

    // Build tool call refs if present.
    var tool_calls: ?[]const local.ToolCallRef = null;
    if (gen_result.tool_calls) |tc| {
        tool_calls = tc;
        gen_result.tool_calls = null; // Transfer ownership to result
    }

    // Duplicate text for the result (gen_result.deinit will free its copy).
    const result_text = allocator_.dupe(u8, gen_result.text) catch return .{ .error_code = @intFromEnum(error_codes.ErrorCode.out_of_memory) };

    return .{
        .text = result_text,
        .token_count = gen_result.generated_tokens,
        .prompt_tokens = gen_result.prompt_tokens,
        .completion_tokens = gen_result.generated_tokens,
        .prefill_ns = gen_result.prefill_ns,
        .generation_ns = gen_result.decode_ns,
        .ttft_ns = 0, // TTFT not tracked in direct generate path
        .error_code = 0,
        .finish_reason = finish_reason,
        .tool_calls = tool_calls,
    };
}

/// Generate with streaming callback using local backend.
pub fn generateStreamingWithBackend(
    allocator_: std.mem.Allocator,
    chat: *Chat,
    parts: []const GenerateContentPart,
    backend: *InferenceBackend,
    config: ?*const CGenerateConfig,
    stream_cb: StreamCallback,
    stream_cb_data: ?*anyopaque,
) GenerateResult {
    // Append user message if parts are provided.
    if (parts.len > 0) {
        appendUserMessageFromParts(chat, parts) catch {
            return .{ .error_code = @intFromEnum(error_codes.ErrorCode.out_of_memory) };
        };
    } else if (chat.conv.len() == 0) {
        return .{ .error_code = @intFromEnum(error_codes.ErrorCode.generation_empty_prompt) };
    }

    switch (backend.backend) {
        .Local => |local_engine| {
            return generateStreamingWithLocalEngine(allocator_, chat, local_engine, config, stream_cb, stream_cb_data);
        },
        .Unspecified => {
            return .{ .error_code = @intFromEnum(error_codes.ErrorCode.invalid_argument) };
        },
    }
}

// =============================================================================
// Public Helpers
// =============================================================================

/// Convert C config to GenerateOptions (basic fields only, no allocations).
///
/// This is a lightweight conversion for the non-streaming API. It copies scalar
/// fields and string slices (which point to C-owned memory). It does NOT
/// tokenize stop sequences or convert logit bias - those require allocation.
pub fn configToGenerateOptions(config: ?*const CGenerateConfig) GenerateOptions {
    var opts = GenerateOptions{};

    const cfg = config orelse return opts;

    if (cfg.max_tokens > 0) opts.max_tokens = cfg.max_tokens;
    if (cfg.max_completion_tokens > 0) opts.max_completion_tokens = cfg.max_completion_tokens;
    if (cfg.max_reasoning_tokens != std.math.maxInt(usize)) opts.max_reasoning_tokens = cfg.max_reasoning_tokens;
    if (cfg.temperature >= 0) opts.temperature = cfg.temperature;
    if (cfg.top_k > 0) opts.top_k = cfg.top_k;
    if (cfg.top_p >= 0) opts.top_p = cfg.top_p;
    if (cfg.min_p >= 0) opts.min_p = cfg.min_p;
    if (cfg.repetition_penalty >= 0) opts.repetition_penalty = cfg.repetition_penalty;
    if (cfg.presence_penalty >= 0) opts.presence_penalty = cfg.presence_penalty;
    if (cfg.frequency_penalty >= 0) opts.frequency_penalty = cfg.frequency_penalty;
    if (cfg.seed != 0) opts.seed = cfg.seed;

    if (cfg.template_override) |t| opts.template_override = std.mem.sliceTo(t, 0);
    if (cfg.extra_context_json) |j| opts.extra_context_json = std.mem.sliceTo(j, 0);
    if (cfg.reasoning_effort) |r| opts.reasoning_effort = std.mem.sliceTo(r, 0);
    if (cfg.tools_json) |t| opts.tools_json = std.mem.sliceTo(t, 0);
    if (cfg.tool_choice) |c| opts.tool_choice = std.mem.sliceTo(c, 0);
    opts.raw_output = cfg.raw_output != 0;
    opts.completions_mode = cfg.completions_mode != 0;

    // Pass through stop flag for cancellation support
    opts.stop_flag = cfg.stop_flag;

    // Pass through prefill progress callback
    opts.prefill_progress_fn = cfg.prefill_progress_fn;
    opts.prefill_progress_data = cfg.prefill_progress_data;

    return opts;
}

// =============================================================================
// Internal Implementation
// =============================================================================

/// Built options with allocated resources that need cleanup.
const BuiltOptions = struct {
    options: GenerateOptions,
    stop_sequences: [][]u32,
    logit_bias: []sampler.LogitBiasEntry,
    external_vision_input: ?vision_types.PrefillVisionInput,

    fn deinit(self: *BuiltOptions, allocator: std.mem.Allocator) void {
        for (self.stop_sequences) |seq| allocator.free(seq);
        if (self.stop_sequences.len > 0) allocator.free(self.stop_sequences);
        if (self.logit_bias.len > 0) allocator.free(self.logit_bias);
        if (self.external_vision_input) |*vision_input| vision_input.deinit(allocator);
    }
};

fn copyExternalVisionInput(
    allocator: std.mem.Allocator,
    cfg: *const CGenerateConfig,
) !?vision_types.PrefillVisionInput {
    if (cfg.external_vision_images == null or cfg.external_vision_image_count == 0) return null;
    if (cfg.external_vision_image_token_id == 0) return error.InvalidArgument;

    const src_images = cfg.external_vision_images.?[0..cfg.external_vision_image_count];
    const images = try allocator.alloc(vision_types.PrefillVisionImage, src_images.len);

    var written: usize = 0;
    errdefer {
        for (images[0..written]) |*img| img.deinit(allocator);
        allocator.free(images);
    }

    for (src_images, 0..) |src_img, i| {
        if (src_img.width == 0 or src_img.height == 0) return error.InvalidArgument;
        if (src_img.token_count == 0) return error.InvalidArgument;
        if (src_img.pixel_count > 0 and src_img.pixels_ptr == null) return error.InvalidArgument;

        const src_pixels: []const f32 = if (src_img.pixel_count > 0)
            src_img.pixels_ptr.?[0..src_img.pixel_count]
        else
            &.{};

        const copied_pixels = try allocator.dupe(f32, src_pixels);
        images[i] = .{
            .pixels = copied_pixels,
            .width = src_img.width,
            .height = src_img.height,
            .grid = .{
                .temporal = src_img.grid_temporal,
                .height = src_img.grid_height,
                .width = src_img.grid_width,
            },
            .token_count = src_img.token_count,
        };
        written += 1;
    }

    return vision_types.PrefillVisionInput{
        .images = images,
        .image_token_id = cfg.external_vision_image_token_id,
    };
}

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
        .external_vision_input = null,
    };

    const cfg = config orelse return result;

    if (cfg.max_tokens > 0) result.options.max_tokens = cfg.max_tokens;
    if (cfg.max_completion_tokens > 0) result.options.max_completion_tokens = cfg.max_completion_tokens;
    if (cfg.max_reasoning_tokens != std.math.maxInt(usize)) result.options.max_reasoning_tokens = cfg.max_reasoning_tokens;
    if (cfg.temperature >= 0) result.options.temperature = cfg.temperature;
    if (cfg.top_k > 0) result.options.top_k = cfg.top_k;
    if (cfg.top_p >= 0) result.options.top_p = cfg.top_p;
    if (cfg.min_p >= 0) result.options.min_p = cfg.min_p;
    if (cfg.repetition_penalty >= 0) result.options.repetition_penalty = cfg.repetition_penalty;
    if (cfg.presence_penalty >= 0) result.options.presence_penalty = cfg.presence_penalty;
    if (cfg.frequency_penalty >= 0) result.options.frequency_penalty = cfg.frequency_penalty;
    if (cfg.seed != 0) result.options.seed = cfg.seed;

    if (cfg.template_override) |t| result.options.template_override = std.mem.sliceTo(t, 0);
    if (cfg.extra_context_json) |j| result.options.extra_context_json = std.mem.sliceTo(j, 0);
    if (cfg.reasoning_effort) |r| result.options.reasoning_effort = std.mem.sliceTo(r, 0);
    if (cfg.tools_json) |t| result.options.tools_json = std.mem.sliceTo(t, 0);
    if (cfg.tool_choice) |c| result.options.tool_choice = std.mem.sliceTo(c, 0);
    result.options.raw_output = cfg.raw_output != 0;

    // Pass through stop flag for cancellation support
    result.options.stop_flag = cfg.stop_flag;

    // Pass through prefill progress callback
    result.options.prefill_progress_fn = cfg.prefill_progress_fn;
    result.options.prefill_progress_data = cfg.prefill_progress_data;

    if (try copyExternalVisionInput(allocator, cfg)) |external_vision_input| {
        result.external_vision_input = external_vision_input;
        result.options.external_vision_input = &result.external_vision_input.?;
    }

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

fn buildOptionsErrorCode(err: anyerror) error_codes.ErrorCode {
    return switch (err) {
        error.InvalidArgument, error.UnsupportedContentType => .invalid_argument,
        else => .out_of_memory,
    };
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
    try std.testing.expectEqual(@as(f32, -1.0), cfg.repetition_penalty);
    try std.testing.expectEqual(@as(f32, -1.0), cfg.presence_penalty);
    try std.testing.expectEqual(@as(f32, -1.0), cfg.frequency_penalty);
    try std.testing.expect(cfg.stop_sequences == null);
    try std.testing.expectEqual(@as(usize, 0), cfg.stop_sequence_count);
    try std.testing.expect(cfg.logit_bias == null);
    try std.testing.expectEqual(@as(usize, 0), cfg.logit_bias_count);
    try std.testing.expectEqual(@as(u64, 0), cfg.seed);
    try std.testing.expect(cfg.template_override == null);
    try std.testing.expect(cfg.extra_context_json == null);
    try std.testing.expect(cfg.extra_body_json == null);
    try std.testing.expectEqual(@as(usize, 0), cfg.raw_output);
    try std.testing.expectEqual(@as(usize, 0), cfg.completions_mode);
    try std.testing.expect(cfg.external_vision_images == null);
    try std.testing.expectEqual(@as(usize, 0), cfg.external_vision_image_count);
    try std.testing.expectEqual(@as(u32, 0), cfg.external_vision_image_token_id);
}

test "configToGenerateOptions completions_mode" {
    var cfg = std.mem.zeroes(CGenerateConfig);
    cfg.completions_mode = 1;
    // Set required defaults for valid config
    cfg.max_reasoning_tokens = std.math.maxInt(usize);
    cfg.temperature = -1.0;
    cfg.top_p = -1.0;
    cfg.min_p = -1.0;
    cfg.repetition_penalty = -1.0;
    cfg.presence_penalty = -1.0;
    cfg.frequency_penalty = -1.0;
    const opts = configToGenerateOptions(&cfg);
    try std.testing.expect(opts.completions_mode);
}

test "configToGenerateOptions completions_mode default off" {
    const opts = configToGenerateOptions(null);
    try std.testing.expect(!opts.completions_mode);
}

test "copyExternalVisionInput copies payload" {
    const allocator = std.testing.allocator;
    const pixels = [_]f32{ 0.25, 0.5, 0.75, 1.0 };
    const c_images = [_]CGenerateVisionImage{
        .{
            .pixels_ptr = pixels[0..].ptr,
            .pixel_count = pixels.len,
            .width = 2,
            .height = 2,
            .grid_temporal = 1,
            .grid_height = 1,
            .grid_width = 1,
            .token_count = 3,
        },
    };
    var cfg: CGenerateConfig = .{};
    cfg.external_vision_images = c_images[0..].ptr;
    cfg.external_vision_image_count = c_images.len;
    cfg.external_vision_image_token_id = 1234;

    var copied = (try copyExternalVisionInput(allocator, &cfg)).?;
    defer copied.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), copied.images.len);
    try std.testing.expectEqual(@as(u32, 1234), copied.image_token_id);
    try std.testing.expectEqual(@as(u32, 2), copied.images[0].width);
    try std.testing.expectEqual(@as(usize, 3), copied.images[0].token_count);
    try std.testing.expectEqual(@as(usize, pixels.len), copied.images[0].pixels.len);
    try std.testing.expectEqual(@as(f32, 0.75), copied.images[0].pixels[2]);
}

test "copyExternalVisionInput rejects missing image token id" {
    const allocator = std.testing.allocator;
    const pixels = [_]f32{1.0};
    const c_images = [_]CGenerateVisionImage{
        .{
            .pixels_ptr = pixels[0..].ptr,
            .pixel_count = pixels.len,
            .width = 1,
            .height = 1,
            .grid_temporal = 1,
            .grid_height = 1,
            .grid_width = 1,
            .token_count = 1,
        },
    };
    var cfg: CGenerateConfig = .{};
    cfg.external_vision_images = c_images[0..].ptr;
    cfg.external_vision_image_count = c_images.len;
    cfg.external_vision_image_token_id = 0;

    try std.testing.expectError(error.InvalidArgument, copyExternalVisionInput(allocator, &cfg));
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
// The following functions require a real LocalEngine and cannot be unit tested
// without a model. Their signatures are verified here; behavior is covered
// through integration tests in bindings/python/tests/.
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
