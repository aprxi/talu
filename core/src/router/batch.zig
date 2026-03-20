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
const error_context = @import("../error_context.zig");

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

/// Maximum number of typed text segments emitted from one decoded chunk after
/// reasoning-tag filtering.
const MAX_FILTER_SEGMENTS = 64;

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
    /// Stable output buffer for filterReasoningTags results. Lives in the
    /// RequestState so returned slices remain valid for the caller's scope.
    filter_out_buf: [MAX_TOKEN_LEN + MAX_TAG_LEN]u8 = undefined,
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

        // Build effective template context (reasoning effort → enable_thinking + tools).
        const effective_context = buildEffectiveContext(self.engine.allocator, opts) catch |err| {
            error_context.setContext("buildEffectiveContext failed: {s}", .{@errorName(err)});
            return err;
        };
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
            error_context.setContext("applyChatTemplate failed: {s}", .{@errorName(err)});
            return err;
        };
        defer self.engine.allocator.free(prompt);

        {
            const tail_len = @min(prompt.len, 200);
            const tail_start = prompt.len - tail_len;
            log.info("batch", "submit", .{
                .prompt_len = prompt.len,
                .prompt_tail = prompt[tail_start..],
                .effective_context = effective_context orelse "(null)",
                .temperature = opts.temperature orelse -99.0,
                .top_k = opts.top_k orelse 999999,
                .max_tokens = opts.max_tokens orelse 0,
            });
        }

        // Detect reasoning start state.
        starts_in_reasoning = chat_template.promptEndsWithReasoningTag(prompt, opts.reasoning_tag);
        state.starts_in_reasoning = starts_in_reasoning;

        // Validate prompt is valid UTF-8 before encoding (diagnostic for Utf8ProcFailed).
        if (!std.unicode.utf8ValidateSlice(prompt)) {
            // Find first invalid byte position.
            var bad_pos: usize = 0;
            var vi: usize = 0;
            while (vi < prompt.len) {
                const seq_len = std.unicode.utf8ByteSequenceLength(prompt[vi]) catch {
                    bad_pos = vi;
                    break;
                };
                if (vi + seq_len > prompt.len) {
                    bad_pos = vi;
                    break;
                }
                if (!std.unicode.utf8ValidateSlice(prompt[vi .. vi + seq_len])) {
                    bad_pos = vi;
                    break;
                }
                vi += seq_len;
            } else {
                bad_pos = vi;
            }
            // Report the byte value and surrounding context as decimal bytes.
            const b0 = prompt[bad_pos];
            const b1: u8 = if (bad_pos + 1 < prompt.len) prompt[bad_pos + 1] else 0;
            const b2: u8 = if (bad_pos + 2 < prompt.len) prompt[bad_pos + 2] else 0;
            const b3: u8 = if (bad_pos + 3 < prompt.len) prompt[bad_pos + 3] else 0;
            error_context.setContext("invalid UTF-8 at byte {d}/{d}, bytes=[0x{X:0>2},0x{X:0>2},0x{X:0>2},0x{X:0>2}]", .{ bad_pos, prompt.len, b0, b1, b2, b3 });
            return error.EncodeFailed;
        }

        // Tokenize prompt.
        const encoded = self.engine.tok.encode(prompt) catch |err| {
            const tok_msg = self.engine.tok.lastError() orelse "unknown";
            error_context.setContext("tok.encode: {s}, prompt_len={d}, detail={s}", .{ @errorName(err), prompt.len, tok_msg });
            return err;
        };
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
            // Mark as tool generation for post-processing (finish reason +
            // tool call extraction). No constrained sampler — the model
            // generates freely in its native tool call format (e.g. Qwen3.5
            // uses <tool_call> XML tags). Parsing happens after generation.
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
                        const now_ns = std.time.nanoTimestamp();
                        const elapsed_ns: i128 = if (now_ns > state.start_ns)
                            now_ns - state.start_ns
                        else
                            0;
                        events_out[event_count] = .{
                            .request_id = raw.request_id,
                            .event_type = .completed,
                            .item_type = @intFromEnum(ItemType.message),
                            .content_type = @intFromEnum(ContentType.output_text),
                            .is_final = true,
                            .text = "",
                            .token_id = raw.token,
                            .tokens_generated = state.engine_token_count,
                            .timestamp_ns = elapsed_ns,
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

            if (state.engine_token_count <= 20) {
                log.info("batch", "token", .{
                    .n = state.engine_token_count,
                    .id = raw.token,
                    .text = decoded_raw,
                    .is_final = raw.is_final,
                });
            }

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
            // A decoded chunk can cross reasoning boundaries (e.g.
            // "reasoning</think>answer"), so we emit per-segment typed deltas
            // instead of assigning one type to the whole chunk.
            var filter_segments: [MAX_FILTER_SEGMENTS]FilteredSegment = undefined;
            var filtered_text: []const u8 = valid;
            var segments: []const FilteredSegment = &.{};
            if (!state.raw_output) {
                const filter_result = filterReasoningTags(state, valid, &filter_segments);
                filtered_text = filter_result.text;
                segments = filter_result.segments;
            } else if (valid.len > 0) {
                filter_segments[0] = .{
                    .start = 0,
                    .len = valid.len,
                    .filter_state = .normal,
                };
                segments = filter_segments[0..1];
            }

            var final_types = classifySegment(state, state.filter_state);

            // Accumulate text once, then emit per-segment events.
            if (filtered_text.len > 0) {
                try state.text_buf.appendSlice(allocator, filtered_text);
                const delta_base = state.delta_buf.items.len;
                try state.delta_buf.appendSlice(allocator, filtered_text);

                for (segments) |seg| {
                    if (seg.len == 0) continue;
                    const seg_types = classifySegment(state, seg.filter_state);
                    final_types = seg_types;
                    if (event_count < events_out.len) {
                        const now_ns = std.time.nanoTimestamp();
                        const elapsed_ns: i128 = if (now_ns > state.start_ns)
                            now_ns - state.start_ns
                        else
                            0;
                        events_out[event_count] = .{
                            .request_id = raw.request_id,
                            .event_type = .text_delta,
                            .item_type = seg_types.item_type,
                            .content_type = seg_types.content_type,
                            .is_final = false,
                            .text = state.delta_buf.items[delta_base + seg.start .. delta_base + seg.start + seg.len],
                            .token_id = raw.token,
                            .tokens_generated = state.engine_token_count,
                            .timestamp_ns = elapsed_ns,
                        };
                        event_count += 1;
                    }
                }
            }

            // Handle completion.
            if (raw.is_final) {
                // Determine finish reason.
                const finish_reason = self.determineFinishReason(state);
                try self.completeRequest(raw.request_id, state, finish_reason);

                if (event_count < events_out.len) {
                    const now_ns = std.time.nanoTimestamp();
                    const elapsed_ns: i128 = if (now_ns > state.start_ns)
                        now_ns - state.start_ns
                    else
                        0;
                    events_out[event_count] = .{
                        .request_id = raw.request_id,
                        .event_type = .completed,
                        .item_type = final_types.item_type,
                        .content_type = final_types.content_type,
                        .is_final = true,
                        .text = "",
                        .token_id = raw.token,
                        .tokens_generated = state.engine_token_count,
                        .timestamp_ns = elapsed_ns,
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
        // Always attempt parsing when is_tool_generation — not just when grammar
        // formally completed. The grammar may not reach .complete (e.g. max_tokens
        // hit) but the generated text may still contain a valid tool call JSON.
        var tool_calls: ?[]CToolCallRef = null;
        var actual_finish_reason = finish_reason;
        if (state.is_tool_generation) {
            if (text) |generated_text| {
                tool_calls = parseToolCalls(generated_text) catch |err| blk: {
                    log.warn("batch", "parseToolCalls failed for tool generation", .{
                        .err = @errorName(err),
                        .text_len = generated_text.len,
                        .finish_reason = @tagName(finish_reason),
                    });
                    break :blk null;
                };
                if (tool_calls != null and finish_reason != .tool_calls) {
                    actual_finish_reason = .tool_calls;
                }
            }
        }

        result.* = .{
            .prompt_tokens = state.prompt_tokens,
            .completion_tokens = state.engine_token_count,
            .prefill_ns = ttft_ns, // Approximation: ttft ≈ prefill.
            .generation_ns = generation_ns,
            .ttft_ns = ttft_ns,
            .finish_reason = actual_finish_reason,
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

/// Typed text segment emitted by reasoning-tag filtering.
const FilteredSegment = struct {
    start: usize,
    len: usize,
    filter_state: FilterState,
};

/// Result of reasoning tag filtering.
const FilterResult = struct {
    /// Text to emit (may be empty if all text was tag content).
    text: []const u8,
    /// Per-segment filter states for `text`.
    segments: []const FilteredSegment,
};

fn classifySegment(
    state: *const RequestState,
    filter_state: FilterState,
) struct { item_type: u8, content_type: u8 } {
    if (state.raw_output) {
        return .{
            .item_type = @intFromEnum(ItemType.message),
            .content_type = @intFromEnum(ContentType.output_text),
        };
    }
    if (filter_state == .reasoning) {
        return .{
            .item_type = @intFromEnum(ItemType.reasoning),
            .content_type = @intFromEnum(ContentType.reasoning_text),
        };
    }
    if (state.is_tool_generation) {
        return .{
            .item_type = @intFromEnum(ItemType.function_call),
            .content_type = @intFromEnum(ContentType.output_text),
        };
    }
    return .{
        .item_type = @intFromEnum(ItemType.message),
        .content_type = @intFromEnum(ContentType.output_text),
    };
}

fn appendFilteredSegment(
    filtered_buf: []u8,
    filtered_len: *usize,
    segments_out: []FilteredSegment,
    segment_count: *usize,
    text: []const u8,
    filter_state: FilterState,
) void {
    if (text.len == 0) return;
    const avail = filtered_buf.len - filtered_len.*;
    if (avail == 0) return;

    const copy_len = @min(text.len, avail);
    const start = filtered_len.*;
    @memcpy(filtered_buf[start..][0..copy_len], text[0..copy_len]);
    filtered_len.* += copy_len;

    if (segment_count.* > 0) {
        var last = &segments_out[segment_count.* - 1];
        if (last.filter_state == filter_state and last.start + last.len == start) {
            last.len += copy_len;
            return;
        }
    }

    if (segment_count.* < segments_out.len) {
        segments_out[segment_count.*] = .{
            .start = start,
            .len = copy_len,
            .filter_state = filter_state,
        };
        segment_count.* += 1;
    }
}

/// Run reasoning tag filter on decoded text. Updates state inline.
/// Returns the text to emit (tag bytes are consumed, not emitted).
///
/// Simplified from iterator.zig filterAndPush: instead of pushing to
/// a ring buffer, we return the filtered text. The caller accumulates.
fn filterReasoningTags(
    state: *RequestState,
    decoded: []const u8,
    segments_out: []FilteredSegment,
) FilterResult {
    // Use state-owned buffer so returned slices remain valid after return.
    const filtered_buf = &state.filter_out_buf;
    var filtered_len: usize = 0;
    var segment_count: usize = 0;

    var i: usize = 0;
    while (i < decoded.len) {
        const byte = decoded[i];

        // Non-ASCII byte: not part of ASCII tag markers.
        if (byte >= 0x80) {
            // Flush any pending tag-match buffer as content.
            if (state.filter_partial_len > 0) {
                appendFilteredSegment(
                    filtered_buf,
                    &filtered_len,
                    segments_out,
                    &segment_count,
                    state.filter_partial_buf[0..state.filter_partial_len],
                    state.filter_state,
                );
                state.filter_partial_len = 0;
            }

            // Forward entire non-ASCII run.
            const run_start = i;
            while (i < decoded.len and decoded[i] >= 0x80) : (i += 1) {}
            const run = decoded[run_start..i];
            state.swallow_next_newline = false;
            appendFilteredSegment(
                filtered_buf,
                &filtered_len,
                segments_out,
                &segment_count,
                run,
                state.filter_state,
            );
            continue;
        }

        // ASCII byte: run tag-matching state machine.
        if (state.filter_partial_len >= MAX_TAG_LEN) {
            // Overflow: flush as literal.
            appendFilteredSegment(
                filtered_buf,
                &filtered_len,
                segments_out,
                &segment_count,
                state.filter_partial_buf[0..state.filter_partial_len],
                state.filter_state,
            );
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
            appendFilteredSegment(
                filtered_buf,
                &filtered_len,
                segments_out,
                &segment_count,
                state.filter_partial_buf[0..state.filter_partial_len],
                state.filter_state,
            );
            state.filter_partial_len = 0;
        }

        i += 1;
    }

    return .{
        .text = filtered_buf[0..filtered_len],
        .segments = segments_out[0..segment_count],
    };
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

    // Normalize flat-format tools to nested OpenAI format for the template.
    const normalized_tools: ?[]const u8 = if (opts.tools_json) |tj|
        try tool_schema_mod.normalizeToolsJson(alloc, tj)
    else
        null;
    defer if (normalized_tools) |nt| alloc.free(nt);
    const tools_fragment: ?[]const u8 = if (normalized_tools) |nt|
        try std.fmt.allocPrint(alloc, ", \"tools\": {s}", .{nt})
    else
        null;
    defer if (tools_fragment) |tf| alloc.free(tf);

    if (opts.extra_context_json) |existing| {
        const trimmed = std.mem.trimRight(u8, existing, " \t\n\r");
        if (trimmed.len > 0 and trimmed[trimmed.len - 1] == '}') {
            const inner = std.mem.trimRight(u8, trimmed[0 .. trimmed.len - 1], " \t\n\r");
            const separator: []const u8 = if (inner.len > 1) ", " else " ";
            const result = try std.fmt.allocPrint(alloc, "{s}{s}\"enable_thinking\": {s}{s} }}", .{
                inner, separator, val, tools_fragment orelse "",
            });
            return @as(?[]const u8, result);
        }
        return null;
    }

    const result = try std.fmt.allocPrint(alloc, "{{\"enable_thinking\": {s}{s}}}", .{
        val, tools_fragment orelse "",
    });
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

test "filterReasoningTags splits reasoning and response in one chunk" {
    var state = RequestState.init();
    defer state.deinit();
    state.filter_state = .reasoning;

    var segments: [MAX_FILTER_SEGMENTS]FilteredSegment = undefined;
    const result = filterReasoningTags(&state, "abc</think>\nXYZ", &segments);

    try std.testing.expectEqualStrings("abcXYZ", result.text);
    try std.testing.expectEqual(@as(usize, 2), result.segments.len);
    try std.testing.expectEqual(@as(usize, 0), result.segments[0].start);
    try std.testing.expectEqual(@as(usize, 3), result.segments[0].len);
    try std.testing.expectEqual(FilterState.reasoning, result.segments[0].filter_state);
    try std.testing.expectEqual(@as(usize, 3), result.segments[1].start);
    try std.testing.expectEqual(@as(usize, 3), result.segments[1].len);
    try std.testing.expectEqual(FilterState.normal, result.segments[1].filter_state);
    try std.testing.expectEqual(FilterState.normal, state.filter_state);
}

test "filterReasoningTags preserves alternating normal and reasoning segments" {
    var state = RequestState.init();
    defer state.deinit();

    var segments: [MAX_FILTER_SEGMENTS]FilteredSegment = undefined;
    const result = filterReasoningTags(&state, "P<think>R</think>A", &segments);

    try std.testing.expectEqualStrings("PRA", result.text);
    try std.testing.expectEqual(@as(usize, 3), result.segments.len);
    try std.testing.expectEqual(FilterState.normal, result.segments[0].filter_state);
    try std.testing.expectEqualStrings("P", result.text[result.segments[0].start .. result.segments[0].start + result.segments[0].len]);
    try std.testing.expectEqual(FilterState.reasoning, result.segments[1].filter_state);
    try std.testing.expectEqualStrings("R", result.text[result.segments[1].start .. result.segments[1].start + result.segments[1].len]);
    try std.testing.expectEqual(FilterState.normal, result.segments[2].filter_state);
    try std.testing.expectEqualStrings("A", result.text[result.segments[2].start .. result.segments[2].start + result.segments[2].len]);
}

/// Parse tool calls from generated text into C-compatible refs.
///
/// Delegates to tool_schema_mod.parseToolCallsFromText which handles
/// both XML format (Qwen3.5) and JSON format.
fn parseToolCalls(generated_text: []const u8) ![]CToolCallRef {
    const parsed_calls = try tool_schema_mod.parseToolCallsFromText(allocator, generated_text);
    defer {
        for (parsed_calls) |*c| {
            var call = c.*;
            call.deinit(allocator);
        }
        allocator.free(parsed_calls);
    }

    const refs = try allocator.alloc(CToolCallRef, parsed_calls.len);
    errdefer allocator.free(refs);

    for (parsed_calls, 0..) |pc, i| {
        const call_id_slice = try tool_schema_mod.generateCallId(allocator);
        defer allocator.free(call_id_slice);

        var ref = std.mem.zeroes(CToolCallRef);
        ref.item_index = @intCast(i);
        ref.call_id = allocator.dupeZ(u8, call_id_slice) catch null;
        ref.name = allocator.dupeZ(u8, pc.name) catch null;
        ref.arguments = allocator.dupeZ(u8, pc.arguments) catch null;
        refs[i] = ref;
    }

    return refs;
}
