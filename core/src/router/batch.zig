//! Batch — Responses-aware batch generation for continuous batching.
//!
//! Wraps the low-level GenericScheduler with per-request state for the full
//! /v1/responses feature surface: reasoning tag filtering, grammar-constrained
//! sampling (tools/structured output), UTF-8 streaming decode, timing, and
//! finish reason determination.
//!
//! The scheduler is single-threaded. The caller drives the step loop. All
//! requests share a single scheduler instance bound to one engine.
//!
//! Lifecycle: create → (submit | cancel | step)* → destroy
//!
//! Thread safety: NOT thread-safe. Caller must serialize all calls.

const std = @import("std");
const local_mod = @import("local.zig");
const capi_bridge = @import("capi_bridge.zig");
const tool_schema_mod = @import("tool_schema.zig");
const commit_mod = @import("commit.zig");
const responses_mod = @import("../responses/root.zig");
const protocol = @import("protocol/root.zig");
const inference = @import("../inference/root.zig");
const inference_types = inference.types;
const FinishReason = inference_types.FinishReason;
const sampler_mod = inference.sampling;
const validate_mod = @import("../validate/root.zig");
const ConstrainedSampler = validate_mod.sampler.ConstrainedSampler;
const GrammarConfig = validate_mod.sampler.GrammarConfig;
const gen_config_mod = @import("../inference/config/generation.zig");
const chat_template = @import("../template/chat_template.zig");
const backend_root = @import("../inference/backend/root.zig");
const Backend = backend_root.Backend;
const log = @import("../log.zig");

const LocalEngine = local_mod.LocalEngine;
const BackendScheduler = local_mod.BackendScheduler;
const SchedulerConfig = local_mod.SchedulerConfig;
const Chat = responses_mod.Chat;
const ItemType = responses_mod.ItemType;
const ContentType = responses_mod.ContentType;

const GenerateOptions = local_mod.GenerateOptions;
const CGenerateConfig = capi_bridge.CGenerateConfig;
const GenerateContentPart = capi_bridge.GenerateContentPart;
const CFinishReason = capi_bridge.CFinishReason;
const CToolCallRef = capi_bridge.CToolCallRef;

const allocator = std.heap.c_allocator;

// =============================================================================
// Constants
// =============================================================================

/// Maximum decoded bytes per token (matches iterator.zig).
const MAX_TOKEN_LEN = 512;

/// Maximum tag length for reasoning filter (matches iterator.zig).
const MAX_TAG_LEN = 64;

/// Maximum text accumulation per request (64 KiB delta buffer).
const MAX_DELTA_BUF = 65536;

// =============================================================================
// Event Types
// =============================================================================

/// Event type for batch events.
pub const EventType = enum(u8) {
    /// Decoded text delta.
    text_delta = 0,
    /// Request completed (final event).
    completed = 1,
    /// Request failed with error.
    err = 2,
};

/// Rich batch event with decoded text and metadata.
pub const BatchEvent = struct {
    request_id: u64,
    event_type: EventType,
    item_type: u8,
    content_type: u8,
    is_final: bool,
    /// Decoded text delta. Slice into per-request delta_buf.
    /// Valid until the next step() call.
    text: []const u8,
    token_id: u32,
    tokens_generated: usize,
    timestamp_ns: i128,
};

/// Completion result for a finished request.
pub const BatchResult = struct {
    prompt_tokens: usize,
    completion_tokens: usize,
    prefill_ns: u64,
    generation_ns: u64,
    ttft_ns: u64,
    finish_reason: CFinishReason,
    /// Full generated text (owned, caller must free via freeResult).
    text: ?[]u8,
    /// Tool calls (owned, caller must free via freeResult).
    tool_calls: ?[]CToolCallRef,

    pub fn deinit(self: *BatchResult) void {
        if (self.text) |t| allocator.free(t);
        if (self.tool_calls) |calls| {
            for (calls) |call| {
                if (call.call_id) |cid| freeZ(cid);
                if (call.name) |n| freeZ(n);
                if (call.arguments) |a| freeZ(a);
            }
            allocator.free(calls);
        }
    }
};

fn freeZ(ptr: [*:0]const u8) void {
    const span = std.mem.span(ptr);
    allocator.free(span.ptr[0 .. span.len + 1]);
}

// =============================================================================
// Per-Request State
// =============================================================================

/// Filter state for reasoning tag parsing.
const FilterState = enum { normal, reasoning };

/// Per-request state for token processing.
const RequestState = struct {
    // --- Decode context ---
    decode_context_token: ?u32 = null,
    utf8_pending: [3]u8 = .{ 0, 0, 0 },
    utf8_pending_len: u8 = 0,

    // --- Reasoning filter ---
    filter_state: FilterState = .normal,
    filter_partial_buf: [MAX_TAG_LEN]u8 = undefined,
    filter_partial_len: u8 = 0,
    swallow_next_newline: bool = false,
    start_marker: []const u8 = "<think>",
    end_marker: []const u8 = "</think>",

    // --- Generation state ---
    is_tool_generation: bool = false,
    raw_output: bool = false,
    starts_in_reasoning: bool = false,
    engine_token_count: usize = 0,

    // --- Text accumulation ---
    /// Full generated text (grows over lifetime of request).
    text_buf: std.ArrayList(u8) = .empty,
    /// Per-step delta text (reset each step, pointed to by BatchEvent.text).
    delta_buf: std.ArrayList(u8) = .empty,

    // --- Grammar ---
    grammar_sampler: ?*ConstrainedSampler = null,
    grammar_schema: ?[]u8 = null,

    // --- Timing ---
    start_ns: i128,
    first_token_ns: i128 = 0,

    // --- Counters ---
    prompt_tokens: usize = 0,

    // --- Owned data for scheduler (prompt tokens, eos copies) ---
    owned_prompt_tokens: ?[]u32 = null,
    owned_eos_tokens: ?[]u32 = null,
    owned_thinking_end_tokens: ?[]u32 = null,

    // --- Stop flag ---
    stop_flag: ?*const std.atomic.Value(bool) = null,

    fn init() RequestState {
        return .{
            .start_ns = std.time.nanoTimestamp(),
        };
    }

    fn deinit(self: *RequestState) void {
        self.text_buf.deinit(allocator);
        self.delta_buf.deinit(allocator);
        if (self.grammar_sampler) |gs| {
            gs.deinit();
            allocator.destroy(gs);
        }
        if (self.grammar_schema) |schema| allocator.free(schema);
        if (self.owned_prompt_tokens) |t| allocator.free(t);
        if (self.owned_eos_tokens) |t| allocator.free(t);
        if (self.owned_thinking_end_tokens) |t| allocator.free(t);
    }
};

// =============================================================================
// BatchWrapper
// =============================================================================

/// Responses-aware batch handle wrapping the low-level scheduler.
///
/// Adds per-request state for decoding, reasoning filtering, grammar
/// constraints, and result assembly. The caller drives the step loop.
pub const BatchWrapper = struct {
    scheduler: BackendScheduler,
    engine: *LocalEngine,
    requests: std.AutoHashMap(u64, *RequestState),
    completed_results: std.AutoHashMap(u64, *BatchResult),

    pub fn init(engine: *LocalEngine, config: SchedulerConfig) !BatchWrapper {
        var scheduler = try engine.createScheduler(config);
        errdefer scheduler.deinit();

        return .{
            .scheduler = scheduler,
            .engine = engine,
            .requests = std.AutoHashMap(u64, *RequestState).init(allocator),
            .completed_results = std.AutoHashMap(u64, *BatchResult).init(allocator),
        };
    }

    pub fn deinit(self: *BatchWrapper) void {
        // Clean up any remaining request states.
        var req_iter = self.requests.valueIterator();
        while (req_iter.next()) |state_ptr| {
            state_ptr.*.deinit();
            allocator.destroy(state_ptr.*);
        }
        self.requests.deinit();

        // Clean up any unclaimed results.
        var res_iter = self.completed_results.valueIterator();
        while (res_iter.next()) |result_ptr| {
            result_ptr.*.deinit();
            allocator.destroy(result_ptr.*);
        }
        self.completed_results.deinit();

        self.scheduler.deinit();
    }

    /// Submit a generation request.
    ///
    /// Applies the chat template, tokenizes the prompt, sets up grammar
    /// constraints (if tools provided), and submits to the scheduler.
    /// Returns a request ID (>0) on success, or error.
    pub fn submit(
        self: *BatchWrapper,
        chat: *Chat,
        config: ?*const CGenerateConfig,
    ) !u64 {
        var state = try allocator.create(RequestState);
        errdefer allocator.destroy(state);
        state.* = RequestState.init();
        errdefer state.deinit();

        // Build GenerateOptions from C config.
        const opts = capi_bridge.configToGenerateOptions(config);

        // Check for raw output mode.
        state.raw_output = opts.raw_output;

        // Pass through stop flag.
        state.stop_flag = opts.stop_flag;

        // Check whether prompt ends with reasoning tag (for filter init).
        var starts_in_reasoning: bool = false;

        // Render conversation to JSON for template application.
        const messages_json = try protocol.completions.serialize(
            self.engine.allocator,
            chat.conv,
            .{ .image_content_type = .image },
        );
        defer self.engine.allocator.free(messages_json);

        // Build effective template context (reasoning effort → enable_thinking).
        const effective_context = try buildEffectiveContext(self.engine.allocator, opts);
        defer if (effective_context) |ctx| self.engine.allocator.free(ctx);

        // Apply chat template.
        const prompt = gen_config_mod.applyChatTemplateWithOverrides(
            self.engine.allocator,
            self.engine.model_path,
            messages_json,
            true,
            opts.template_override,
            effective_context,
        ) catch |err| {
            log.warn("batch", "Chat template failed", .{ .err = @errorName(err) });
            return err;
        };
        defer self.engine.allocator.free(prompt);

        // Detect reasoning start state.
        starts_in_reasoning = chat_template.promptEndsWithReasoningTag(prompt, opts.reasoning_tag);
        state.starts_in_reasoning = starts_in_reasoning;

        // Tokenize prompt.
        const encoded = try self.engine.tok.encode(prompt);
        defer self.engine.allocator.free(encoded);

        // Prepend BOS if configured.
        const bos_token_id: ?u32 = if (self.engine.gen_config.bos_token_id) |id|
            (if (self.engine.gen_config.add_bos_token) id else null)
        else if (self.engine.loaded.config.bos_token_id) |id|
            (if (self.engine.gen_config.add_bos_token and id >= 0) @intCast(id) else null)
        else
            null;

        var prepend_bos = bos_token_id != null;
        if (prepend_bos and encoded.len > 0 and encoded[0] == bos_token_id.?) {
            prepend_bos = false;
        }

        const prompt_tokens = if (prepend_bos) blk: {
            const tokens = try allocator.alloc(u32, encoded.len + 1);
            tokens[0] = bos_token_id.?;
            @memcpy(tokens[1..], encoded);
            break :blk tokens;
        } else try allocator.dupe(u32, encoded);

        state.owned_prompt_tokens = prompt_tokens;
        state.prompt_tokens = prompt_tokens.len;

        // Determine max_tokens.
        const base_max_tokens = if (opts.max_completion_tokens != null and opts.max_tokens == null) blk: {
            const raw_budget = if (opts.max_reasoning_tokens) |mrt|
                mrt
            else
                maxThinkingTokensForEffort(opts.reasoning_effort);
            break :blk raw_budget + opts.max_completion_tokens.?;
        } else opts.max_tokens orelse chat.max_tokens;

        const grammar_slack: usize = 64;

        // Set up grammar sampler for tools.
        const use_tools = opts.tools_json != null and
            (opts.tool_choice == null or !std.mem.eql(u8, opts.tool_choice.?, "none"));

        if (use_tools) {
            state.grammar_schema = try tool_schema_mod.toolsToGrammarSchema(
                allocator,
                opts.tools_json.?,
            );
            const gs = try allocator.create(ConstrainedSampler);
            errdefer allocator.destroy(gs);
            gs.* = try ConstrainedSampler.init(
                allocator,
                state.grammar_schema.?,
                GrammarConfig{},
                self.engine.gen_config.eos_token_ids,
                null,
                null,
            );
            state.grammar_sampler = gs;
            state.is_tool_generation = true;
        }

        const effective_grammar = state.grammar_sampler orelse chat.grammar_sampler;

        const max_tokens = if (effective_grammar != null and base_max_tokens > 0)
            base_max_tokens + grammar_slack
        else
            base_max_tokens;

        // Build sampling config (mirrors local.zig generateFromPrompt).
        const temperature = opts.temperature orelse chat.temperature;
        const top_k = opts.top_k orelse chat.top_k;
        const top_p = opts.top_p orelse chat.top_p;
        const min_p = opts.min_p orelse chat.min_p;
        const repetition_penalty = opts.repetition_penalty orelse chat.repetition_penalty;
        const presence_penalty = opts.presence_penalty orelse 0.0;
        const frequency_penalty = opts.frequency_penalty orelse 0.0;

        var sampling_config = sampler_mod.SamplingConfig{
            .strategy = .greedy,
            .logit_bias = opts.logit_bias,
            .seed = opts.seed,
            .presence_penalty = presence_penalty,
            .frequency_penalty = frequency_penalty,
        };
        if (temperature > 0 and (self.engine.gen_config.do_sample or opts.temperature != null)) {
            sampling_config = .{
                .strategy = .top_k,
                .temperature = temperature,
                .top_k = top_k,
                .top_p = top_p,
                .min_p = min_p,
                .repetition_penalty = repetition_penalty,
                .presence_penalty = presence_penalty,
                .frequency_penalty = frequency_penalty,
                .logit_bias = opts.logit_bias,
                .seed = opts.seed,
            };
        }

        // Thinking budget.
        const raw_thinking_budget = if (opts.max_reasoning_tokens) |mrt|
            mrt
        else
            maxThinkingTokensForEffort(opts.reasoning_effort);
        const answer_reserve = if (opts.max_completion_tokens) |mct|
            mct
        else
            @max(@as(usize, 256), max_tokens / 4);
        const thinking_budget = if (raw_thinking_budget > 0)
            @min(raw_thinking_budget, max_tokens -| answer_reserve)
        else
            @as(usize, 0);

        // Tokenize thinking end sequence.
        const thinking_end_tokens = if (thinking_budget > 0)
            self.engine.tok.encode("</think>\n\n") catch null
        else
            null;
        if (thinking_end_tokens) |t| {
            state.owned_thinking_end_tokens = allocator.dupe(u32, t) catch null;
            self.engine.allocator.free(t);
        }

        // Submit to scheduler.
        const request_id = try self.scheduler.submit(prompt_tokens, max_tokens, .{
            .eos_token_ids = self.engine.gen_config.eos_token_ids,
            .sampling = sampling_config,
            .grammar_sampler = effective_grammar,
            .stop_flag = opts.stop_flag,
            .max_thinking_tokens = thinking_budget,
            .thinking_end_tokens = if (state.owned_thinking_end_tokens) |t| t else &.{},
            .max_completion_tokens = opts.max_completion_tokens orelse 0,
        });

        try self.requests.put(request_id, state);
        return request_id;
    }

    /// Run one generation step. Returns events in the caller-provided buffer.
    ///
    /// Each event contains decoded text, item/content type metadata, and timing.
    /// Text pointers in events are valid until the next step() call.
    pub fn step(self: *BatchWrapper, events_out: []BatchEvent) !usize {
        // Clear per-step delta buffers for all active requests.
        var req_iter = self.requests.valueIterator();
        while (req_iter.next()) |state_ptr| {
            state_ptr.*.delta_buf.clearRetainingCapacity();
        }

        // Run one decode step on the scheduler.
        const raw_events = try self.scheduler.step();
        defer allocator.free(raw_events);

        var event_count: usize = 0;

        for (raw_events) |raw| {
            const state = self.requests.get(raw.request_id) orelse continue;

            // Skip EOS tokens (don't decode them).
            if (gen_config_mod.isEosToken(self.engine.gen_config.eos_token_ids, raw.token)) {
                if (raw.is_final) {
                    try self.completeRequest(raw.request_id, state, .eos_token);
                    if (event_count < events_out.len) {
                        events_out[event_count] = .{
                            .request_id = raw.request_id,
                            .event_type = .completed,
                            .item_type = @intFromEnum(ItemType.message),
                            .content_type = @intFromEnum(ContentType.output_text),
                            .is_final = true,
                            .text = "",
                            .token_id = raw.token,
                            .tokens_generated = state.engine_token_count,
                            .timestamp_ns = std.time.nanoTimestamp(),
                        };
                        event_count += 1;
                    }
                }
                continue;
            }

            // Handle reasoning start on first non-EOS token.
            if (state.engine_token_count == 0 and state.starts_in_reasoning) {
                state.filter_state = .reasoning;
            }
            state.engine_token_count += 1;

            // Record first token time.
            if (state.first_token_ns == 0) {
                state.first_token_ns = std.time.nanoTimestamp();
            }

            // Decode token to raw bytes with context.
            const decoded_raw = try decodeTokenWithContext(self.engine, state, raw.token);
            defer self.engine.allocator.free(decoded_raw);

            if (decoded_raw.len > 0) {
                state.decode_context_token = raw.token;
            }

            // UTF-8 assembly (same algorithm as iterator.zig).
            var combined_buf: [3 + MAX_TOKEN_LEN]u8 = undefined;
            const pending_len: usize = state.utf8_pending_len;
            @memcpy(combined_buf[0..pending_len], state.utf8_pending[0..pending_len]);
            const raw_copy_len = @min(decoded_raw.len, combined_buf.len - pending_len);
            @memcpy(combined_buf[pending_len..][0..raw_copy_len], decoded_raw[0..raw_copy_len]);
            const total_len = pending_len + raw_copy_len;
            const combined = combined_buf[0..total_len];

            const valid_end = utf8ValidPrefix(combined);
            const valid = combined[0..valid_end];
            const trailing = combined[valid_end..total_len];

            // Store trailing incomplete UTF-8 bytes.
            state.utf8_pending_len = 0;
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
                    for (trailing[1..]) |cb| {
                        if (cb & 0xC0 != 0x80) {
                            all_valid = false;
                            break;
                        }
                    }
                    if (all_valid) {
                        state.utf8_pending_len = @intCast(trailing.len);
                        @memcpy(state.utf8_pending[0..trailing.len], trailing);
                    }
                }
            }

            if (valid.len == 0 and !raw.is_final) continue;

            // Classify text via reasoning filter / tool mode.
            var item_type: u8 = @intFromEnum(ItemType.message);
            var content_type: u8 = @intFromEnum(ContentType.output_text);
            var emit_text: []const u8 = valid;

            if (state.is_tool_generation) {
                item_type = @intFromEnum(ItemType.function_call);
                content_type = @intFromEnum(ContentType.output_text);
            } else if (state.raw_output) {
                // Raw mode: no filtering.
            } else {
                // Reasoning tag filter: classify and strip tags.
                const filter_result = filterReasoningTags(state, valid);
                emit_text = filter_result.text;
                if (state.filter_state == .reasoning) {
                    item_type = @intFromEnum(ItemType.reasoning);
                    content_type = @intFromEnum(ContentType.reasoning_text);
                }
            }

            // Accumulate text.
            if (emit_text.len > 0) {
                try state.text_buf.appendSlice(allocator, emit_text);
                const delta_start = state.delta_buf.items.len;
                try state.delta_buf.appendSlice(allocator, emit_text);

                if (event_count < events_out.len) {
                    events_out[event_count] = .{
                        .request_id = raw.request_id,
                        .event_type = .text_delta,
                        .item_type = item_type,
                        .content_type = content_type,
                        .is_final = false,
                        .text = state.delta_buf.items[delta_start..],
                        .token_id = raw.token,
                        .tokens_generated = state.engine_token_count,
                        .timestamp_ns = std.time.nanoTimestamp(),
                    };
                    event_count += 1;
                }
            }

            // Handle completion.
            if (raw.is_final) {
                // Determine finish reason.
                const finish_reason = self.determineFinishReason(state);
                try self.completeRequest(raw.request_id, state, finish_reason);

                if (event_count < events_out.len) {
                    events_out[event_count] = .{
                        .request_id = raw.request_id,
                        .event_type = .completed,
                        .item_type = item_type,
                        .content_type = content_type,
                        .is_final = true,
                        .text = "",
                        .token_id = raw.token,
                        .tokens_generated = state.engine_token_count,
                        .timestamp_ns = std.time.nanoTimestamp(),
                    };
                    event_count += 1;
                }
            }
        }

        // Drain scheduler's completed queue.
        const completed = self.scheduler.popCompleted();
        if (completed.len > 0) allocator.free(completed);

        return event_count;
    }

    /// Cancel a request.
    pub fn cancel(self: *BatchWrapper, request_id: u64) bool {
        const cancelled = self.scheduler.cancel(request_id);
        if (cancelled) {
            self.scheduler.remove(request_id);
            if (self.requests.fetchRemove(request_id)) |kv| {
                var state = kv.value;
                state.deinit();
                allocator.destroy(state);
            }
            const completed = self.scheduler.popCompleted();
            if (completed.len > 0) allocator.free(completed);
        }
        return cancelled;
    }

    /// Take the completion result for a finished request.
    ///
    /// Returns null if the request is not yet complete or was already taken.
    /// Caller owns the result and must call result.deinit() to free.
    pub fn takeResult(self: *BatchWrapper, request_id: u64) ?*BatchResult {
        if (self.completed_results.fetchRemove(request_id)) |kv| {
            return kv.value;
        }
        return null;
    }

    /// Check if any requests are active or pending.
    pub fn hasActive(self: *const BatchWrapper) bool {
        return self.scheduler.hasActive();
    }

    /// Get count of active requests.
    pub fn activeCount(self: *const BatchWrapper) usize {
        return self.scheduler.activeCount();
    }

    // =========================================================================
    // Internal
    // =========================================================================

    fn determineFinishReason(self: *const BatchWrapper, state: *const RequestState) CFinishReason {
        _ = self;
        // Tool call: grammar completed.
        if (state.is_tool_generation) {
            if (state.grammar_sampler) |gs| {
                if (gs.state == .complete) return .tool_calls;
            }
        }
        // Cancellation via stop flag.
        if (state.stop_flag) |flag| {
            if (flag.load(.acquire)) return .cancelled;
        }
        // Default: EOS.
        return .eos_token;
    }

    fn completeRequest(
        self: *BatchWrapper,
        request_id: u64,
        state: *RequestState,
        finish_reason: CFinishReason,
    ) !void {
        const now = std.time.nanoTimestamp();
        const ttft_ns: u64 = if (state.first_token_ns > 0)
            @intCast(state.first_token_ns - state.start_ns)
        else
            0;
        const generation_ns: u64 = @intCast(now - state.start_ns);

        // Build result.
        const result = try allocator.create(BatchResult);
        errdefer allocator.destroy(result);

        // Copy accumulated text.
        const text = if (state.text_buf.items.len > 0)
            try allocator.dupe(u8, state.text_buf.items)
        else
            null;
        errdefer if (text) |t| allocator.free(t);

        // Parse tool calls if this was a tool generation.
        var tool_calls: ?[]CToolCallRef = null;
        if (finish_reason == .tool_calls) {
            if (text) |generated_text| {
                tool_calls = parseToolCalls(generated_text) catch null;
            }
        }

        result.* = .{
            .prompt_tokens = state.prompt_tokens,
            .completion_tokens = state.engine_token_count,
            .prefill_ns = ttft_ns, // Approximation: ttft ≈ prefill.
            .generation_ns = generation_ns,
            .ttft_ns = ttft_ns,
            .finish_reason = finish_reason,
            .text = text,
            .tool_calls = tool_calls,
        };

        try self.completed_results.put(request_id, result);

        // Remove request from active tracking and clean up.
        self.scheduler.remove(request_id);
        if (self.requests.fetchRemove(request_id)) |kv| {
            var req_state = kv.value;
            req_state.deinit();
            allocator.destroy(req_state);
        }
    }
};

// =============================================================================
// Helpers (private)
// =============================================================================

/// Decode a token to raw bytes using context-aware pair decoding.
/// Same algorithm as iterator.zig decodeRawWithContext.
fn decodeTokenWithContext(engine: *LocalEngine, state: *const RequestState, token_id: u32) ![]u8 {
    if (state.decode_context_token) |ctx_token| {
        const ctx_raw = engine.tok.decodeRawBytes(
            &[_]u32{ctx_token},
            .{ .skip_special_tokens = true },
        ) catch return engine.tok.decodeRawBytes(
            &[_]u32{token_id},
            .{ .skip_special_tokens = true },
        );
        defer engine.allocator.free(ctx_raw);

        const pair_raw = engine.tok.decodeRawBytes(
            &[_]u32{ ctx_token, token_id },
            .{ .skip_special_tokens = true },
        ) catch return engine.tok.decodeRawBytes(
            &[_]u32{token_id},
            .{ .skip_special_tokens = true },
        );
        errdefer engine.allocator.free(pair_raw);

        const prefix = longestCommonPrefixLen(ctx_raw, pair_raw);
        if (prefix < ctx_raw.len) {
            engine.allocator.free(pair_raw);
            return engine.tok.decodeRawBytes(
                &[_]u32{token_id},
                .{ .skip_special_tokens = true },
            );
        }

        const delta = pair_raw[prefix..];
        if (delta.len == pair_raw.len) {
            engine.allocator.free(pair_raw);
            return engine.tok.decodeRawBytes(
                &[_]u32{token_id},
                .{ .skip_special_tokens = true },
            );
        }

        const out = try engine.allocator.alloc(u8, delta.len);
        @memcpy(out, delta);
        engine.allocator.free(pair_raw);
        return out;
    }

    return engine.tok.decodeRawBytes(
        &[_]u32{token_id},
        .{ .skip_special_tokens = true },
    );
}

/// Result of reasoning tag filtering.
const FilterResult = struct {
    /// Text to emit (may be empty if all text was tag content).
    text: []const u8,
};

/// Run reasoning tag filter on decoded text. Updates state inline.
/// Returns the text to emit (tag bytes are consumed, not emitted).
///
/// Simplified from iterator.zig filterAndPush: instead of pushing to
/// a ring buffer, we return the filtered text. The caller accumulates.
fn filterReasoningTags(state: *RequestState, decoded: []const u8) FilterResult {
    // We accumulate filtered output into state.delta_buf temporarily.
    // But since the caller also writes to delta_buf, we use a local buffer.
    var filtered_buf: [MAX_TOKEN_LEN + MAX_TAG_LEN]u8 = undefined;
    var filtered_len: usize = 0;

    var i: usize = 0;
    while (i < decoded.len) {
        const byte = decoded[i];

        // Non-ASCII byte: not part of ASCII tag markers.
        if (byte >= 0x80) {
            // Flush any pending tag-match buffer as content.
            if (state.filter_partial_len > 0) {
                const plen: usize = state.filter_partial_len;
                const avail = filtered_buf.len - filtered_len;
                const copy_len = @min(plen, avail);
                @memcpy(filtered_buf[filtered_len..][0..copy_len], state.filter_partial_buf[0..copy_len]);
                filtered_len += copy_len;
                state.filter_partial_len = 0;
            }

            // Forward entire non-ASCII run.
            const run_start = i;
            while (i < decoded.len and decoded[i] >= 0x80) : (i += 1) {}
            const run = decoded[run_start..i];
            state.swallow_next_newline = false;
            const avail = filtered_buf.len - filtered_len;
            const copy_len = @min(run.len, avail);
            @memcpy(filtered_buf[filtered_len..][0..copy_len], run[0..copy_len]);
            filtered_len += copy_len;
            continue;
        }

        // ASCII byte: run tag-matching state machine.
        if (state.filter_partial_len >= MAX_TAG_LEN) {
            // Overflow: flush as literal.
            const plen: usize = state.filter_partial_len;
            const avail = filtered_buf.len - filtered_len;
            const copy_len = @min(plen, avail);
            @memcpy(filtered_buf[filtered_len..][0..copy_len], state.filter_partial_buf[0..copy_len]);
            filtered_len += copy_len;
            state.filter_partial_len = 0;
        }
        state.filter_partial_buf[state.filter_partial_len] = byte;
        state.filter_partial_len += 1;

        const buf = state.filter_partial_buf[0..state.filter_partial_len];

        // Complete marker check.
        if (state.filter_state == .normal and std.mem.eql(u8, buf, state.start_marker)) {
            state.filter_partial_len = 0;
            state.filter_state = .reasoning;
            state.swallow_next_newline = false;
            i += 1;
            continue;
        }
        if (state.filter_state == .reasoning and std.mem.eql(u8, buf, state.end_marker)) {
            state.filter_partial_len = 0;
            state.filter_state = .normal;
            state.swallow_next_newline = true;
            i += 1;
            continue;
        }

        // Prefix check.
        const is_prefix = switch (state.filter_state) {
            .normal => std.mem.startsWith(u8, state.start_marker, buf),
            .reasoning => std.mem.startsWith(u8, state.end_marker, buf),
        };

        if (!is_prefix) {
            // Not a tag prefix: flush buffer as content.
            if (state.swallow_next_newline) {
                state.swallow_next_newline = false;
                if (state.filter_partial_len == 1 and buf[0] == '\n') {
                    state.filter_partial_len = 0;
                    i += 1;
                    continue;
                }
            }
            const plen: usize = state.filter_partial_len;
            const avail = filtered_buf.len - filtered_len;
            const copy_len = @min(plen, avail);
            @memcpy(filtered_buf[filtered_len..][0..copy_len], state.filter_partial_buf[0..copy_len]);
            filtered_len += copy_len;
            state.filter_partial_len = 0;
        }

        i += 1;
    }

    // Return a slice of the valid decoded text from the input.
    // We must return a stable pointer — use a subset of the input or
    // write into the delta_buf. Since we wrote into filtered_buf (stack),
    // we need to copy to a stable location. We reuse the delta_buf for this.
    if (filtered_len == 0) return .{ .text = "" };

    // Since we're called from step() which already uses delta_buf,
    // and the text will be appended to delta_buf by the caller,
    // we return a view into our stack buffer. The caller copies immediately.
    // Actually, let's return a slice from decoded if no filtering happened.
    if (filtered_len == decoded.len and std.mem.eql(u8, filtered_buf[0..filtered_len], decoded)) {
        return .{ .text = decoded };
    }

    // Filtering modified the text. We need to return it from a stable location.
    // Write into the state's text_buf as a temporary, since the caller will
    // copy via appendSlice anyway.
    // Actually, we need a different approach: return the stack data and have
    // the caller handle it. Since `decoded` is on the stack too (combined_buf),
    // and the caller immediately copies, this is fine — both are stack-valid
    // for the duration of the step() iteration body.
    return .{ .text = filtered_buf[0..filtered_len] };
}

/// Build effective template context (reasoning effort → enable_thinking).
/// Same logic as local.zig buildEffectiveContext (duplicated to avoid
/// making the private function public).
fn buildEffectiveContext(alloc: std.mem.Allocator, opts: GenerateOptions) !?[]const u8 {
    const enable_thinking: bool = if (opts.max_reasoning_tokens) |mrt|
        mrt > 0
    else if (opts.reasoning_effort) |effort|
        !std.mem.eql(u8, effort, "none")
    else
        true;

    const val: []const u8 = if (enable_thinking) "true" else "false";

    if (opts.extra_context_json) |existing| {
        const trimmed = std.mem.trimRight(u8, existing, " \t\n\r");
        if (trimmed.len > 0 and trimmed[trimmed.len - 1] == '}') {
            const inner = std.mem.trimRight(u8, trimmed[0 .. trimmed.len - 1], " \t\n\r");
            const separator: []const u8 = if (inner.len > 1) ", " else " ";
            const result = try std.fmt.allocPrint(alloc, "{s}{s}\"enable_thinking\": {s} }}", .{
                inner, separator, val,
            });
            return @as(?[]const u8, result);
        }
        return null;
    }

    const result = try std.fmt.allocPrint(alloc, "{{\"enable_thinking\": {s}}}", .{val});
    return @as(?[]const u8, result);
}

/// Map reasoning effort level to a max thinking token budget.
fn maxThinkingTokensForEffort(effort: ?[]const u8) usize {
    const e = effort orelse return 4096;
    if (std.mem.eql(u8, e, "none")) return 0;
    if (std.mem.eql(u8, e, "low")) return 512;
    if (std.mem.eql(u8, e, "medium")) return 4096;
    if (std.mem.eql(u8, e, "high")) return 16384;
    if (std.mem.eql(u8, e, "xhigh")) return 32768;
    return 4096;
}

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
        else
            0;

        if (seq_len == 0) {
            // Invalid lead byte.
            break;
        }
        if (i + seq_len > bytes.len) {
            // Incomplete sequence.
            break;
        }
        // Verify continuations.
        var valid = true;
        var j: usize = 1;
        while (j < seq_len) : (j += 1) {
            if (bytes[i + j] & 0xC0 != 0x80) {
                valid = false;
                break;
            }
        }
        if (!valid) break;
        i += seq_len;
        last_valid = i;
    }
    return last_valid;
}

fn longestCommonPrefixLen(a: []const u8, b: []const u8) usize {
    const n = @min(a.len, b.len);
    var i: usize = 0;
    while (i < n and a[i] == b[i]) : (i += 1) {}
    return i;
}

/// Parse tool call JSON into C-compatible tool call refs.
fn parseToolCalls(generated_text: []const u8) ![]CToolCallRef {
    const parsed = try tool_schema_mod.parseToolCall(allocator, generated_text);
    defer {
        allocator.free(parsed.name);
        allocator.free(parsed.arguments);
    }

    const calls = try allocator.alloc(CToolCallRef, 1);
    errdefer allocator.free(calls);

    // Generate a call_id for the tool call.
    const call_id_slice = try tool_schema_mod.generateCallId(allocator);
    defer allocator.free(call_id_slice);

    var call = std.mem.zeroes(CToolCallRef);
    call.item_index = 0;
    call.call_id = allocator.dupeZ(u8, call_id_slice) catch null;
    call.name = allocator.dupeZ(u8, parsed.name) catch null;
    call.arguments = allocator.dupeZ(u8, parsed.arguments) catch null;
    calls[0] = call;

    return calls;
}
